import torch
import torchvision
from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import transforms
from PIL import Image
#import randaugmentMIDL as ra
import randaugment_new_ranges as ra

import os
import numpy as np
from torch.utils.data import Dataset

class Camelyon17(Dataset):
    """Custom dataset for reading a datasets of .npy arrays with patches."""

    def __init__(self, x_path, y_path, transform=None, randaugment=False, rand_m=5, rand_n=3, randomize=None,randaugment_transforms_set = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x_path = x_path
        self.y_path = y_path
        self.rand_n = rand_n
        self.rand_m = rand_m
        self.randaugment_transforms_set = randaugment_transforms_set
        self.randaugment = randaugment
        self.randomize = randomize
        self.x = np.load(self.x_path)
        self.targets = np.load(self.y_path)
        print('Dataset length:',len(self.targets))
        if len(np.unique(self.targets))>2:
            print('loaded data has', len(np.unique(self.targets)),'classes')
             
            self.targets[self.targets==0 ]=0
            self.targets[self.targets==1 ]=1
            self.targets[self.targets==2 ]=0
            self.targets[self.targets==3 ]=1
            self.targets[self.targets==4 ]=1
            self.targets[self.targets==5 ]=1
            self.targets[self.targets==6 ]=1

            print('modified to ', len(np.unique(self.targets)),'classes')
        self.transform = transform
        self.n_classes = 2
        print('In the dataloader. Loaded the images.')

    def __len__(self):
        print('Length of the dataset:',len(self.targets))
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.patches= (self.x[idx, ...])
        self.y=self.targets[idx]

        if self.randaugment:
                self.x_augmented = ra.distort_image_with_randaugment(image = self.patches, num_layers = self.rand_n, magnitude = self.rand_m,  randomize=self.randomize,randaugment_transforms_set=self.randaugment_transforms_set)
        else:
            self.x_augmented = self.patches


        if self.transform is not None:
            self.x_augmented = self.transform(self.x_augmented)
        else:
            self.x_augmented = self.x_augmented
       
        self.x_augmented = np.moveaxis(self.x_augmented, -1,0).astype(np.float32)/255.0
        #To make labels categorical
        #self.y = np.eye(self.n_classes)[self.y].astype(np.long)
        #print('Maximum value of the images passed to the network :',np.max(self.x_augmented))
        return self.x_augmented, self.y.astype(np.long)




def num_class(dataset):
    return {
        'cifar10': 10,
        'camelyon17': 2,
        'tiger': 2,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'reduced_cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]

def get_dataloaders(dataset, batch, num_workers, dataroot, train_set, val_set, randaugment=False, rand_m=5, rand_n=3, randomize=False,randaugment_transforms_set=None):
    
    

    if dataset == 'camelyon17':
        #Load train data
        x_path=dataroot+train_set+'x.npy'
        y_path=dataroot+train_set+'y.npy'

        val_x_path = dataroot+val_set+'x.npy'
        val_y_path = dataroot+val_set+'y.npy'
        #Load train data
        total_trainset = Camelyon17(x_path, y_path, transform=None,randaugment=randaugment, rand_m=rand_m, rand_n=rand_n, randomize=randomize, randaugment_transforms_set=randaugment_transforms_set) # 4000 trainset
        total_valset = Camelyon17(val_x_path, val_y_path, transform=None,randaugment=False, rand_m=None, rand_n=None, randomize=randomize,randaugment_transforms_set=None)


    elif dataset == 'tiger':
        #Load train data
        x_path=dataroot+train_set+'x.npy'
        y_path=dataroot+train_set+'y.npy'

        val_x_path = dataroot+val_set+'x.npy'
        val_y_path = dataroot+val_set+'y.npy'
        #Load train data
        total_trainset = Camelyon17(x_path, y_path, transform=None,randaugment=randaugment, rand_m=rand_m, rand_n=rand_n, randomize=randomize, randaugment_transforms_set=randaugment_transforms_set) # 4000 trainset
        total_valset = Camelyon17(val_x_path, val_y_path, transform=None,randaugment=False, rand_m=None, rand_n=None, randomize=randomize,randaugment_transforms_set=None)
    else:
        raise ValueError('invalid dataset name=%s' % dataset)


    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True, drop_last=False,
        pin_memory=True, num_workers=num_workers)

    validloader = torch.utils.data.DataLoader(
        total_valset, batch_size=batch, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=num_workers)

    return trainloader, validloader
