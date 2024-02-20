import os
import numpy as np
import torch
import shutil
import os.path
import pandas as pd
import torchvision.transforms as transf
import matplotlib.pyplot as plt


class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res=correct_k.mul_(100.0/batch_size)
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    if not os.path.exists(os.path.join(path, 'scripts')):
       os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

#----------------------------------------------------------------------------------------------------

def history_to_file(io_path, epoch, lr, train_obj, train_metric, valid_obj, valid_metric):
    if os.path.isfile(io_path): 
        df = pd.read_csv(io_path)
        #print('file exists')
    else:
        df = pd.DataFrame(columns=['epoch','train_obj','train_metric','valid_obj','valid_metric'])
        #print('creating')
    #print(df)   
    #print('lr',lr,'train_obj',train_obj,'train_metric',train_metric,'valid_obj',valid_obj)
    df = df.append({'lr': lr,'epoch': epoch,'train_obj':train_obj,'train_metric': train_metric,'valid_obj': valid_obj,'valid_metric': valid_metric}, ignore_index=True).to_csv(io_path,index=False)
#----------------------------------------------------------------------------------------------------

def log_to_file(io_path, test_set,  n, m, valid_obj, valid_metric):
    if os.path.isfile(io_path): 
        df = pd.read_csv(io_path)
        #print('file exists')
    else:
        df = pd.DataFrame(columns=['test_set',' n','m','valid_obj','valid_metric'])
        #print('creating')
    #print(df)   
    #print('lr',lr,'train_obj',train_obj,'train_metric',train_metric,'valid_obj',valid_obj)
    df = df.append({'test_set': test_set,' n': n,'m':m,'valid_obj': valid_obj,'valid_metric': valid_metric}, ignore_index=True).to_csv(io_path,index=False)
#----------------------------------------------------------------------------------------------------
def log_sample_pred_label_to_file(io_path, prediction, label):
    if os.path.isfile(io_path): 
        df = pd.read_csv(io_path)
        #print('file exists')
    else:
        df = pd.DataFrame(columns=['prediction','label'])
        #print('creating')
    #print(df)   
    #print('lr',lr,'train_obj',train_obj,'train_metric',train_metric,'valid_obj',valid_obj)
    df = df.append({'prediction': prediction,'label': label}, ignore_index=True).to_csv(io_path,index=False)

#----------------------------------------------------------------------------------------------------

def log_sample_pred_label_to_file_a(io_path, prediction, label, dataset):
  import pandas as pd

  #create DataFrame
  df = pd.DataFrame({'dataset': dataset,
                     'label': list(label),
                     'prediction': list(prediction)})
  df.to_csv(io_path, mode='a', index=False, header=False)
#----------------------------------------------------------------------------------------------------
def plot_history(history_path,  plot_path):
    """
    Makes a plot of training history. Top is a logplot with loss values. The learning rate is also plotted in top using
    the right axis. Bottom is a plot with metrics.

    Args:
        history_path (str): path to CSV file containing the training history (see HistoryCsv).
        loss_list (list): list of losses to plot.
        metric_list (list): list of metrics to plot.
        plot_path (str): output path to store PNG with history plot.
    """

    # Format data
    df = pd.read_csv(history_path)

    # Subplots: losses and metrics
    fig, (ax_loss, ax_metric) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot learning rate in right axis
    if 'lr' in df.columns:
        ax_lr = ax_loss.twinx()
        lines = ax_lr.semilogy(df.index.values, df.loc[:, 'lr'], '-k', label='lr')
        ax_lr.set_ylabel('Learning rate')
    else:
        lines = None

    # Plot learning rate in right axis
    if 'train_obj' in df.columns:
      try:
        new_lines = ax_loss.semilogy(df.index.values, df.loc[:, 'train_obj'], '-c', label='training loss')
        #ax_loss.set_ylabel('Training objective')
        if lines:
              lines += new_lines
        else:
              lines = new_lines
      except:
          pass

    #print('lines',df.index.values, df.loc[:, 'train_obj'])
    # Plot learning rate in right axis    
    if 'valid_obj' in df.columns:
      try:
        new_lines = ax_loss.semilogy(df.index.values, df.loc[:, 'valid_obj'], '-m', label='validation loss')
        #ax_loss.set_ylabel('Validation objective')
        if lines:
                lines += new_lines
        else:
                lines = new_lines
      except:
          pass
   # Final touches
    ax_loss.legend()
    ax_loss.grid()
    ax_loss.set_title('Training Summary')
    ax_loss.set_ylabel('Loss')
    lines_labels = [l.get_label() for l in lines]
    ax_loss.legend(lines, lines_labels, loc=0)


    if 'train_metric' in df.columns:
      try:
        new_lines = ax_metric.plot(df.index.values, df.loc[:, 'train_metric'], '-c', label='training accuracy')
        #ax_loss.set_ylabel('Training metric')
        if lines:
                lines += new_lines
        else:
                lines = new_lines
      except:
        pass
 

    # Plot learning rate in right axis    
    if 'valid_metric' in df.columns:
      try:
        new_lines = ax_metric.plot(df.index.values, df.loc[:, 'valid_metric'], '-m', label='validation accuracy')
        #ax_loss.set_ylabel('Validation metric')
        if lines:
                lines += new_lines
        else:
              lines = new_lines
      except:
          pass
    '''
    # Plot learning rate in right axis    
    if 'epoch' in df.columns:
      try:
        new_lines = ax_metric.semilogy(df.index.values, df.loc[:, 'epoch'], '-k', label='epoch')
        #ax_loss.set_ylabel('Validation metric')
        if lines:
                lines += new_lines
        else:
                lines = new_lines
      except:
          pass'''

    
 


    # Final touches
    ax_metric.legend()
    ax_metric.grid()
    ax_metric.set_ylabel('Metric')
    ax_metric.set_xlabel('Number of epochs')
    ax_metric.legend(loc=0)

    # Store plot in disk
    plt.savefig(plot_path)
    plt.close()

#----------------------------------------------------------------------------------------------------

