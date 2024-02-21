import copy

import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score
from torch import nn


def accuracy_topk(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res=correct_k.mul_(100.0/batch_size)
  return res

def accuracy(output, target,threshold=0.5):
    """Computes the precision@k for the specified values of k"""
    target_arr=target.detach().cpu().numpy().astype('int')
    output_arr=output.detach().cpu().numpy()[:,1]
    w = compute_label_weights(target_arr, one_hot=False)
    #output_arr = np.argmax(output_arr, axis=-1)
    #print('target_arr.shape',target_arr.shape)
    #print('output_arr.shape',output_arr.shape)
    output_arr = np.array(output_arr > threshold).astype('int')
    
    res = accuracy_score(target_arr, output_arr, sample_weight=w)
    
    return res

def compute_label_weights(y_true, one_hot=True):

    if one_hot:
        y_true_single = np.argmax(y_true, axis=-1)
    else:
        y_true_single = y_true

    w = np.ones(y_true_single.shape[0])
    for idx, i in enumerate(np.bincount(y_true_single)):
        w[y_true_single == idx] *= 1/(i / float(y_true_single.shape[0]))

    return w


# ----------------------------------------------------------------------------------------------------

def accuracy_weighted_fn(y_true, y_pred, label_map=None, threshold=0.5):

    # Label map
    if label_map is not None:
        # y_true = map_probabilities(y_true, label_map)
        y_pred = map_probabilities(y_pred, label_map)

    # Weights
    w = compute_label_weights(y_true, one_hot=True)

    # Thresholding
    y_pred_th = np.array(y_pred > threshold).astype('int')

    # No one-hot
    y_true = np.argmax(y_true, axis=-1)
    y_pred_th = np.argmax(y_pred_th, axis=-1)

    # # Label map
    # if label_map is not None:
    #     y_pred_th = np.array([label_map[str(pred)] for pred in y_pred_th])

    # Score
    acc = accuracy_score(y_true, y_pred_th, sample_weight=w)

    return acc

class CrossEntropyLabelSmooth(torch.nn.Module):
    def __init__(self, num_classes, epsilon, reduction='mean'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        log_probs = self.logsoftmax(input)
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        if self.epsilon > 0.0:
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        targets = targets.detach()
        loss = (-targets * log_probs)

        if self.reduction in ['avg', 'mean']:
            loss = torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class Accumulator:
    def __init__(self):
        self.metrics = defaultdict(lambda: 0.)

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))

    def __truediv__(self, other):
        newone = Accumulator()
        for key, value in self.items():
            if isinstance(other, str):
                if other != key:
                    newone[key] = value / self[other]
                else:
                    newone[key] = value
            else:
                newone[key] = value / other
        return newone


class SummaryWriterDummy:
    def __init__(self, log_dir):
        pass

    def add_scalar(self, *args, **kwargs):
        pass
