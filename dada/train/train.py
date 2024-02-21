import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

import itertools
import json
import logging
import math
import os
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from tqdm import tqdm
from theconf import Config as C, ConfigArgumentParser
import torch.nn as nn
from common import get_logger, EMA, add_filehandler
from data import get_dataloaders
from metrics import accuracy, accuracy_topk, Accumulator, CrossEntropyLabelSmooth
from networks import get_model, num_class
from tf_port.rmsprop import RMSpropTF
from aug_mixup import CrossEntropyMixUpLabelSmooth, mixup
from warmup_scheduler import GradualWarmupScheduler
from torch.autograd import Variable
import warnings
import wandb
warnings.filterwarnings("ignore")

logger = get_logger('Fast AutoAugment')
logger.setLevel(logging.INFO)
def compute_label_weights(y_true, one_hot=False):

    if one_hot:
        y_true_single = np.argmax(y_true, axis=-1)
    else:
        y_true_single = y_true

    w = np.ones(y_true_single.shape[0])
    for idx, i in enumerate(np.bincount(y_true_single)):
        w[y_true_single == idx] *= 1/(i / float(y_true_single.shape[0]))

    return w

def run_epoch(model, loader, loss_fn, optimizer, desc_default='', epoch=0, writer=None, verbose=1, scheduler=None, is_master=True, ema=None, wd=0.0, tqdm_disabled=False):
    if verbose:
        loader = tqdm(loader, disable=tqdm_disabled)
        #loader.set_description('[%s %04d/%04d]' % (desc_default, epoch, C.get()['epoch']))

    params_without_bn = [params for name, params in model.named_parameters() if not ('_bn' in name or '.bn' in name)]

    loss_ema = 0
    metrics = Accumulator()
    cnt = 0
    total_steps = len(loader)
    steps = 0
    for data, label in loader:
        steps += 1
        data, label = data.cuda(), label.cuda()

        preds = model(data)
        loss = loss_fn(preds, label)
        batch_weight = Variable(torch.Tensor(compute_label_weights(label.detach().cpu().numpy())), requires_grad=False).cuda()
        loss = loss*batch_weight
        loss = torch.mean(loss)
    

        if optimizer:
            #loss += wd * (1. / 2.) * sum([torch.sum(p ** 2) for p in params_without_bn])
            loss.backward()
            grad_clip = C.get()['optimizer'].get('clip', 5.0)
            #if grad_clip > 0:
            #    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()


        acc = accuracy_topk(preds, label)
        metrics.add_dict({
            'loss': loss.item() * len(data),
            'acc': acc.item() * len(data),
            
        })
        cnt += len(data)


        loss_ema = loss_ema+loss.item()


      

        


        del preds, loss, acc, data, label

    loss_ema = loss_ema/steps
    if (desc_default=="valid") and (scheduler is not None):
            scheduler.step(loss_ema)
    
    if tqdm_disabled and verbose:
        if optimizer:
            logger.info('[%s %03d/%03d] %s lr=%.6f', desc_default, epoch, C.get()['epoch'], metrics / cnt, optimizer.param_groups[0]['lr'])
        else:
            logger.info('[%s %03d/%03d] %s', desc_default, epoch, C.get()['epoch'], metrics / cnt)
    
    metrics /= cnt
    if optimizer:
        metrics.metrics['lr'] = optimizer.param_groups[0]['lr']
    if verbose:
        for key, value in metrics.items():
            writer.add_scalar(key, value, epoch)
    return metrics


def train_and_eval(tag, dataroot, test_ratio=0.0, cv_fold=0, reporter=None, metric='last', save_path=None, only_eval=False, local_rank=-1, evaluation_interval=1,use_wandb=False,wandb=None):
    total_batch = C.get()["batch"]
    logger.info('In train_and_eval, local_rank %s' % local_rank)
    
    is_master = local_rank < 0 or dist.get_rank() == 0
    if is_master:
        add_filehandler(logger, os.path.join('projects','autoaugmentation','from_chansey_review','fastautoaugment','FastAutoAugment','master' + '.log'))

    if not reporter:
        reporter = lambda **kwargs: 0

    max_epoch = C.get()['epoch']
    trainsampler, trainloader, validloader, testloader_ = get_dataloaders(C.get()['dataset'], C.get()['batch'], dataroot, test_ratio, split_idx=cv_fold, multinode=(local_rank >= 0))
    logger.info('In train_and_eval, got datasets, local_rank = %s' % local_rank)
    # create a model & an optimizer

    model = get_model(C.get()['model'], num_class(C.get()['dataset']), local_rank=local_rank)

    logger.info('In train_and_eval, got model')
    criterion_ce = criterion = nn.CrossEntropyLoss()#CrossEntropyLabelSmooth(num_class(C.get()['dataset']), C.get().conf.get('lb_smooth', 0))
    
    logger.info('In train_and_eval, got criterion')
    optimizer = torch.optim.Adam(model.parameters(),0.003)
    
    logger.info('In train_and_eval, got opyimizer')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5, eps=1e-08, verbose=True) 
    
   
    logger.info('In train_and_eval, got lr')
    if not tag or not is_master:
        from metrics import SummaryWriterDummy as SummaryWriter
        logger.warning('tag not provided, no tensorboard log.')
    else:
        from tensorboardX import SummaryWriter
    writers = [SummaryWriter(log_dir='./logs/%s/%s' % (tag, x)) for x in ['train', 'valid', 'test']]

    if C.get()['optimizer']['ema'] > 0.0 and is_master:
        # https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4?u=ildoonet
        ema = EMA(C.get()['optimizer']['ema'])
    else:
        ema = None

    result = OrderedDict()
    epoch_start = 1
    


    

    if only_eval:
        logger.info('evaluation only+')
        model.eval()
        rs = dict()
        rs['train'] = run_epoch(model, trainloader, criterion, None, desc_default='train', epoch=0, writer=writers[0], is_master=is_master)

        with torch.no_grad():
            rs['valid'] = run_epoch(model, validloader, criterion, None, desc_default='valid', epoch=0, writer=writers[1], is_master=is_master)
            rs['test'] = run_epoch(model, testloader_, criterion, None, desc_default='*test', epoch=0, writer=writers[2], is_master=is_master)
            '''if ema is not None and len(ema) > 0:
                model_ema.load_state_dict({k.replace('module.', ''): v for k, v in ema.state_dict().items()})
                rs['valid'] = run_epoch(model_ema, validloader, criterion_ce, None, desc_default='valid(EMA)', epoch=0, writer=writers[1], verbose=is_master, tqdm_disabled=tqdm_disabled)
                rs['test'] = run_epoch(model_ema, testloader_, criterion_ce, None, desc_default='*test(EMA)', epoch=0, writer=writers[2], verbose=is_master, tqdm_disabled=tqdm_disabled)'''
        for key, setname in itertools.product(['loss', 'acc'], ['train', 'valid', 'test']):
            if setname not in rs:
                continue
            result['%s_%s' % (key, setname)] = rs[setname][key]
        result['epoch'] = 0
        return result
    
    tqdm_disabled = True #bool(os.environ.get('TASK_NAME', '')) and local_rank != 0  # KakaoBrain Environment
    # train loop
    best_acc = 0
    for epoch in range(epoch_start, max_epoch + 1):
   
        model.train()
        rs = dict()
        print('In training loop')
        rs['train'] = run_epoch(model, trainloader, criterion, optimizer, desc_default='train', epoch=epoch, writer=writers[0], verbose=(is_master and local_rank <= 0), scheduler=scheduler, ema=ema, wd=C.get()['optimizer']['decay'], tqdm_disabled=tqdm_disabled)
        model.eval()

        if math.isnan(rs['train']['loss']):
            raise Exception('train loss is NaN.')

       
        if (epoch % evaluation_interval == 0 or epoch == max_epoch):
            #if is_master and (epoch == max_epoch):
            with torch.no_grad():
                rs['valid'] = run_epoch(model, validloader, criterion_ce, None, desc_default='valid', epoch=epoch, writer=writers[1], verbose=is_master,scheduler=scheduler, tqdm_disabled=tqdm_disabled)
                rs['test'] = run_epoch(model, testloader_, criterion_ce, None, desc_default='*test', epoch=epoch, writer=writers[2], verbose=is_master, tqdm_disabled=tqdm_disabled)
                wandb.log({"train_loss": rs["train"]["loss"]})
                wandb.log({"valid_loss": rs["valid"]["loss"]})
                wandb.log({"train_acc": rs["train"]["acc"]})
                wandb.log({"valid_acc": rs["valid"]["acc"]})
                wandb.log({"lr": optimizer.param_groups[0]['lr']})
                '''if ema is not None:
                    model_ema.load_state_dict({k.replace('module.', ''): v for k, v in ema.state_dict().items()})
                    rs['valid'] = run_epoch(model_ema, validloader, criterion_ce, None, desc_default='valid(EMA)', epoch=epoch, writer=writers[1], verbose=is_master, scheduler=scheduler, tqdm_disabled=tqdm_disabled)
                    rs['test'] = run_epoch(model_ema, testloader_, criterion_ce, None, desc_default='*test(EMA)', epoch=epoch, writer=writers[2], verbose=is_master, tqdm_disabled=tqdm_disabled)'''

                logger.info(
                f'epoch={epoch} '
                f'[train] loss={rs["train"]["loss"]:.4f} acc={rs["train"]["acc"]:.4f} '
                f'[valid] loss={rs["valid"]["loss"]:.4f} acc={rs["valid"]["acc"]:.4f} '
                f'[test] loss={rs["test"]["loss"]:.4f} acc={rs["test"]["acc"]:.4f} '
            )

            if epoch == epoch_start:
                best_loss = rs[metric]['loss']
    
            if metric == 'last' or rs[metric]['loss'] < best_loss:
                if metric != 'last':
                    best_loss = rs[metric]['loss']
                for key, setname in itertools.product(['loss', 'acc'], ['train', 'valid', 'test']):
                    result['%s_%s' % (key, setname)] = rs[setname][key]
                    logger.info('Recorded model perfomance')
                result['epoch'] = epoch

                writers[1].add_scalar('valid_loss/best', rs['valid']['loss'], epoch)
                writers[2].add_scalar('test_loss/best', rs['test']['loss'], epoch)
            
                reporter(
                    loss_valid=rs['valid']['loss'], top1_valid=rs['valid']['acc'],
                    loss_test=rs['test']['loss'], top1_test=rs['test']['acc']
                )
                
                # save checkpoint
                if save_path:
                    logger.info('save model@%d to %s, err=%.4f' % (epoch, save_path, best_loss))
                    torch.save({
                        'epoch': epoch,
                        'log': {
                            'train': rs['train'].get_dict(),
                            'valid': rs['valid'].get_dict(),
                            'test': rs['test'].get_dict(),
                        },
                        'optimizer': optimizer.state_dict(),
                        'model': model.state_dict(),
                        'ema': ema.state_dict() if ema is not None else None,
                    }, save_path)

    del model
    logger.info('Deleted the model from memory')

    result['acc_test'] = best_acc
    return result


if __name__ == '__main__':
    from datetime import datetime

    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='', help='torchvision data folder')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--cv-ratio', type=float, default=0.0)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--evaluation-interval', type=int, default=1)
    parser.add_argument('--only-eval', action='store_true')
    args = parser.parse_args()

    assert (args.only_eval and args.save) or not args.only_eval, 'checkpoint path not provided in evaluation mode.'

    if not args.only_eval:
        if args.save:
            logger.info('checkpoint will be saved at %s' % args.save)
        else:
            logger.warning('Provide --save argument to save the checkpoint. Without it, training result will not be saved!')
    savedir='/data/pathology/projects/autoaugmentation/from_chansey_review/fastautoaugment/models/'+C.get()['val_set']+'_'+'aug_'+C.get()['aug_setup']+'nu2u'+datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    add_filehandler(logger, savedir+'.log' )
    import time
    t = time.time()
    arg_dict={}
    arg_dict['val_set']=C.get()['val_set']
    arg_dict['batch_size']=C.get()['batch']
    arg_dict['learning_rate']=C.get()['lr']
    arg_dict['randomize']=C.get()['val_set']
    arg_dict['aug_setup']=C.get()['aug_setup']
    arg_dict['dataset']=C.get()['dataset']

    project_name='fast-autoaugment'+C.get()['val_set']
    wandb.init(project=project_name, name = C.get()['val_set']+'_'+'aug_'+C.get()['aug_setup']+'nu2u',notes="first logging with wandb", 
  config=arg_dict,dir='/data/pathology/projects/autoaugmentation/from_chansey_review')
    result = train_and_eval(args.tag, args.dataroot, test_ratio=args.cv_ratio, cv_fold=args.cv, save_path=savedir+'.pth', only_eval=args.only_eval, local_rank=args.local_rank, metric='valid', evaluation_interval=args.evaluation_interval, use_wandb=True,wandb=wandb)
    
    elapsed = time.time() - t

    logger.info('done.')
    logger.info('model: %s' % C.get()['model'])
    logger.info('augmentation: %s' % C.get()['aug'])
    logger.info('\n' + json.dumps(result, indent=4))
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('acc error in testset: %.4f' % (1. - result['acc_test']))
    logger.info(args.save)
