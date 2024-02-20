import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import random
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from dataset import get_dataloaders, num_class
from torch.autograd import Variable
from resnet import ResNet
from datetime import datetime
from torchsummary import summary
import wandb 
 
'''
./c-submit --require-gpu-mem="8G" --gpu-count="1" --require-mem="21G" --require-cpus="2" --priority="low" khrystynafaryna 9548 72 doduo1.umcn.nl/khrystynafaryna/kf_container_rand:latest python3 /data/pathology/projects/autoaugmentation/from_chansey_review/randaugment/train.py
./c-submit --require-gpu-mem="8G" --gpu-count="1" --require-mem="30G" --require-cpus="8" --priority="low" khrystynafaryna 9548 72 doduo1.umcn.nl/khrystynafaryna/kf_container_rand:latest python3 /data/pathology/projects/autoaugmentation/from_chansey_review/randaugment/train.py --val_set  val_A2_jb_ --m 21 --n 7 --learning_rate 0.0003
'''
parser = argparse.ArgumentParser("tiger")
parser.add_argument('--dataroot', type=str, default='/data/pathology/projects/autoaugmentation/from_chansey_upd/data/tiger/patches/', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='tiger',choices=['tiger','camelyon17','none'],
                    help='location of the data corpus')
parser.add_argument('--train_set', type=str, default='training_', help='train file name')
parser.add_argument('--val_set', type=str, default='val_A2_jb_', help='val file name')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.003, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.00003, help='min learning rate')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--model', type=str, default='resnet18', help='path to save the model')
parser.add_argument('--save', type=str, default='/data/pathology/projects/autoaugmentation/from_chansey_review/randaugment/experiments/', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--sample_weighted_loss', type=bool, default=True, help="sample weights in loss function")
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--use_cuda', type=bool, default=True, help="use cuda default True")
parser.add_argument('--use_parallel', type=bool, default=False, help="use data parallel default False")
parser.add_argument('--num_workers', type=int, default=8, help="num_workers")
parser.add_argument('--randaugment', type=bool, default=True, help='use randaugment augmentation')
parser.add_argument('--m', type=int, default=5, help="magnitude of randaugment")
parser.add_argument('--n', type=int, default=3, help="number of layers randaugment")
parser.add_argument('--randomize', type=bool, default=True, help="randomize magnitude in randaugment")
parser.add_argument('--randaugment_transforms_set', type=str, default='review',choices=['review','midl2021','original','midl2021_tr2eb','midl2021_trsh2eb'],help='which set of randaugment transforms to use')
parser.add_argument('--lr_schedule', type=str, default='rlop', choices = ['cos','exp','rlop'], help = "which lr scheduler to use")
parser.add_argument('--optimizer_type', type=str, default='adam', choices = ['sdg','adam','rms'], help = "which optimizer to use")
parser.add_argument('--save_best', type=bool, default=True, help="If True, updating model weights only whe minimum of loss occurs, else updating weights every epoch")
parser.add_argument('--project_name', type=str, default='randaugment_new_ranges',choices=['randaugment_new_ranges','randaugment','manual','clean'], help="If True, updating model weights only whe minimum of loss occurs, else updating weights every epoch")

args = parser.parse_args()
arg_dict={}

arg_dict['val_set']=args.val_set
arg_dict['batch_size']=args.batch_size
arg_dict['learning_rate']=args.learning_rate
arg_dict['learning_rate_min']=args.learning_rate_min
arg_dict['randaugment']=args.randaugment
arg_dict['randomize']=args.randomize
arg_dict['m']=args.m
arg_dict['n']=args.n
arg_dict['randaugment_transforms_set']=args.randaugment_transforms_set
arg_dict['optimizer_type']=args.optimizer_type
arg_dict['lr_schedule']=args.lr_schedule
arg_dict['dataset']=args.dataset

project_name=args.project_name+args.val_set[3:-1]

time_stamp= args.dataset+args.val_set+'n_'+str(args.n)+'_m_'+str(args.m)+'_t_'+args.randaugment_transforms_set+'/'+str(datetime.now())
wandb.init(project=project_name, name = args.val_set+'n_'+str(args.n)+'_m_'+str(args.m)+'_t_'+args.randaugment_transforms_set,notes="first logging with wandb", 
  config=arg_dict,dir='/data/pathology/projects/autoaugmentation/from_chansey_review')
args.save = os.path.join(args.save,time_stamp)
os.makedirs(args.save, exist_ok=True)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
sub_policies=None
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

num_classes=2


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  # np.random.seed(args.seed)
  # torch.cuda.set_device(args.gpu)
  # torch.manual_seed(args.seed)
  # torch.cuda.manual_seed(args.seed)
  cudnn.benchmark = True
  
  cudnn.enabled=True
  
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  
  model = ResNet(dataset='imagenet-dropout', depth=18, num_classes=num_classes, bottleneck=True)
  print(model)
  model = model.cuda()
  #summary(model,(3,128,128))

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  if args.optimizer_type=='sdg':
    optimizer = torch.optim.SGD(model.parameters(),args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)
  elif args.optimizer_type=='adam':
    optimizer = torch.optim.Adam(model.parameters(),args.learning_rate)
  elif args.optimizer_type=='rms':
    optimizer = torch.optim.RMSprop(model.parameters(),args.learning_rate,momentum=args.momentum,weight_decay=args.weight_decay)
  else:
    print('This optimizer is not implemented')


  train_queue, valid_queue = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        dataroot=args.dataroot, train_set=args.train_set, val_set=args.val_set, randaugment=args.randaugment, rand_m=args.m, rand_n=args.n, randomize=args.randomize, randaugment_transforms_set=args.randaugment_transforms_set)
  


  if args.lr_schedule=='rlop':
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-5, eps=1e-08, verbose=True)
  elif args.lr_schedule=='cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  else:
    print('This lr schedule is not implemented')

 
  for epoch in range(args.epochs):
    #scheduler.step()
    logging.info('Epoch: %d ', epoch)

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    scheduler.step(valid_obj)
    

    if args.save_best:

      if epoch == 0:
        best_valid_obj=valid_obj.copy()
        utils.save(model, os.path.join(args.save, 'weights.pt'))
      else:

          if valid_obj < best_valid_obj:
            best_valid_obj=valid_obj.copy()
            utils.save(model, os.path.join(args.save, 'weights.pt'))
            logging.info('New best valid_obj achieved %f, overwriting the model  weights...', valid_obj)
          else:
            logging.info('Not updating weights this epoch, valid_obj %f, higher than minimum...', valid_obj)

    else:
      utils.save(model, os.path.join(args.save, 'weights.pt'))
      logging.info('Option save_best is off, updating weights every epoch...')

    wandb.log({"train_loss": train_obj})
    wandb.log({"valid_loss": valid_obj})
    wandb.log({"train_acc": train_acc})
    wandb.log({"valid_acc": valid_acc})
    wandb.log({"lr": optimizer.param_groups[0]['lr']})
    utils.history_to_file(io_path = args.save+"/log.csv", epoch = epoch, lr = optimizer.param_groups[0]['lr'], train_obj = train_obj, train_metric = train_acc, valid_obj = valid_obj, valid_metric = valid_acc)
    utils.plot_history(history_path = args.save+"/log.csv", plot_path = args.save+"/log.png")

       
def train(train_queue, model, criterion, optimizer, sample_weighted_loss=True):

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        
      
        input = Variable(input.type(torch.FloatTensor), requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(non_blocking=True)
        optimizer.zero_grad()
        logits = model(input)
        if sample_weighted_loss:

            loss = criterion(logits, target)

            batch_weight = Variable(torch.Tensor(compute_label_weights(target.detach().cpu().numpy())), requires_grad=False).cuda()
            loss = loss*batch_weight
            loss = torch.mean(loss)

        else:
            loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1 = utils.accuracy(logits.detach(), target.detach())
        n = input.size(0)
        objs.update(loss.detach().cpu().numpy(), n)
        top1.update(prec1.detach().cpu().numpy(), n)
       
        #if step % args.report_freq == 0:
        #    logging.info('train %03d %e %f', step, objs.avg, top1.avg)
        #    #logging.info('train auc: %f',roc_auc_score(target.detach().cpu().numpy(),logits.detach().cpu().numpy()[:,1]) 
    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):

    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):

            input = Variable(input.type(torch.FloatTensor)).cuda()
            target = Variable(target).cuda(non_blocking=True)


            logits = model(input)
            loss = criterion(logits, target)
            batch_weight = Variable(torch.Tensor(compute_label_weights(target.detach().cpu().numpy())), requires_grad=False).cuda()
            loss = loss*batch_weight
            loss = torch.mean(loss)

            prec1 = utils.accuracy(logits, target)
            n = input.size(0)

            objs.update(loss.detach().cpu().numpy(), n)
            top1.update(prec1.detach().cpu().numpy(), n)


            #if step % args.report_freq == 0:
            #    logging.info('valid %03d %e %f', step, objs.avg, top1.avg)
                
    return top1.avg, objs.avg



def compute_label_weights(y_true, one_hot=False):

    if one_hot:
        y_true_single = np.argmax(y_true, axis=-1)
    else:
        y_true_single = y_true

    w = np.ones(y_true_single.shape[0])
    for idx, i in enumerate(np.bincount(y_true_single)):
        w[y_true_single == idx] *= 1/(i / float(y_true_single.shape[0]))

    return w


if __name__ == '__main__':
    print('Entering main...')
    main()



