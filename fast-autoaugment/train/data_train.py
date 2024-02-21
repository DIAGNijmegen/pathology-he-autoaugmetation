import logging

import numpy as np
import os

import math
import random
import torch
import torchvision
from PIL import Image

from torch.utils.data import SubsetRandomSampler, Sampler, Subset, ConcatDataset
import torch.distributed as dist
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C

from archive import arsaug_policy, fa_camelyon17_np5_ns500_nr50_rh_umcu_2, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet, fa_camelyon17_1st,fa_camelyon17_2nd, fa_camelyon17_np5_ns500_nr50, fa_camelyon17_np5_ns500_nr100
from augmentations import *
from common import get_logger
from imagenet import ImageNet
from networks.efficientnet_pytorch.model import EfficientNet

logger = get_logger('Fast AutoAugment')
logger.setLevel(logging.INFO)
_IMAGENET_PCA = {
    'eigval': [0.2175, 0.0188, 0.0045],
    'eigvec': [
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ]
}
_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)



from torch.utils.data import Dataset
class Camelyon17(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x_path, y_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x_path = x_path
        self.y_path = y_path

        self.x = np.load(self.x_path)
        #print('self.x shape:',self.x.shape)
        #if (self.x).shape[-1]==3:
        #    self.x= np.moveaxis(self.x,3,1)
        self.targets = np.load(self.y_path)
        print('In definition length',len(self.targets))
        self.transform = transform
        self.n_classes = 2
       
        print('In the dataloader')

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        self.patches= (self.x[idx, ...])#.astype(np.uint8)
        self.y=(self.targets[idx]).astype(np.long)

        #print('self.x shape __getitem__:',self.patches.shape)
        #print('max before transform', np.max(np.asarray(self.patches)))
        if self.transform:
            self.patches = self.transform(self.patches)
        #print('max after transform', np.max(np.asarray(self.patches)))
        self.patches = (self.patches)/255.0
        #print('self.x shape __getitem__ after transform:',self.patches.shape)
        #self.y = np.eye(self.n_classes)[self.y]#.astype(np.long)

        return self.patches, self.y

def get_dataloaders(dataset, batch, dataroot, split=0.15, split_idx=0, multinode=False, target_lb=-1):
    if 'cifar' in dataset or 'svhn' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    elif 'imagenet' in dataset:
        input_size = 224
        sized_size = 256

        if 'efficientnet' in C.get()['model']['type']:
            input_size = EfficientNet.get_image_size(C.get()['model']['type'])
            sized_size = input_size + 32    # TODO
            # sized_size = int(round(input_size / 224. * 256))
            # sized_size = input_size
            logger.info('size changed to %d/%d.' % (input_size, sized_size))

        transform_train = transforms.Compose([
            EfficientNetRandomCrop(input_size),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            # transforms.RandomResizedCrop(input_size, scale=(0.1, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
            ),
            transforms.ToTensor(),
            Lighting(0.1, _IMAGENET_PCA['eigval'], _IMAGENET_PCA['eigvec']),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            EfficientNetCenterCrop(input_size),
            transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    elif 'camelyon17' in dataset:
        transform_train = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])#[transforms.ToTensor()])

            
        transform_test = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])])#[transforms.ToTensor()])
    else:
        raise ValueError('dataset=%s' % dataset)

    total_aug = augs = None
    if isinstance(C.get()['aug'], list):
        logger.debug('augmentation provided.')
        transform_train.transforms.insert(0 ,Augmentation(C.get()['aug']))
    else:
        logger.debug('augmentation: %s' % C.get()['aug'])
        if C.get()['aug'] == 'fa_reduced_cifar10':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_cifar10()))

        elif C.get()['aug'] == 'fa_reduced_imagenet':
            transform_train.transforms.insert(0, Augmentation(fa_resnet50_rimagenet()))
        elif C.get()['aug'] == 'fa_camelyon17':
            transform_train.transforms.insert(0, Augmentation(fa_camelyon17_np5_ns500_nr50_rh_umcu_2()))

        elif C.get()['aug'] == 'fa_reduced_svhn':
            transform_train.transforms.insert(0, Augmentation(fa_reduced_svhn()))

        elif C.get()['aug'] == 'arsaug':
            transform_train.transforms.insert(0, Augmentation(arsaug_policy()))
        elif C.get()['aug'] == 'autoaug_cifar10':
            transform_train.transforms.insert(0, Augmentation(autoaug_paper_cifar10()))
        elif C.get()['aug'] == 'autoaug_extend':
            transform_train.transforms.insert(0, Augmentation(autoaug_policy()))
        elif C.get()['aug'] in ['default']:
            pass
        else:
            raise ValueError('not found augmentations. %s' % C.get()['aug'])

    if C.get()['cutout'] > 0:
        transform_train.transforms.append(CutoutDefault(C.get()['cutout']))

    

    elif dataset == 'imagenet':
        total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=transform_train, download=True)
        testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]
    elif dataset == 'camelyon17':
        x_path='/data/pathology/projects/autoaugmentation/from_chansey_upd/data/lymph/patches/training_x.npy'
        y_path='/data/pathology/projects/autoaugmentation/from_chansey_upd/data/lymph/patches/training_y.npy'

        val_x_path='/data/pathology/projects/autoaugmentation/from_chansey_upd/data/lymph/patches/val_rh_umcu_x.npy'
        val_y_path='/data/pathology/projects/autoaugmentation/from_chansey_upd/data/lymph/patches/val_rh_umcu_y.npy'
        #val_x_path='/mnt/netcache/pathology/projects/autoaugmentation/data/lymph/patches/test_lpe_x.npy'
        #val_y_path='/mnt/netcache/pathology/projects/autoaugmentation/data/lymph/patches/test_lpe_y.npy'

        test_x_path='/data/pathology/projects/autoaugmentation/from_chansey_upd/data/lymph/patches/val_rh_umcu_x.npy'
        test_y_path='/data/pathology/projects/autoaugmentation/from_chansey_upd/data/lymph/patches/val_rh_umcu_y.npy'
        #test_x_path='/mnt/netcache/pathology/projects/autoaugmentation/data/lymph/patches/test_umcu_x.npy'
        #test_y_path='/mnt/netcache/pathology/projects/autoaugmentation/data/lymph/patches/test_umcu_y.npy'
        total_trainset = Camelyon17(x_path, y_path, transform=transform_train) # 4000 trainset
        total_valset = Camelyon17(val_x_path, val_y_path, transform=transform_train)
        total_testset = Camelyon17(test_x_path, test_y_path, transform=transform_test)
    elif dataset == 'reduced_imagenet':
        # randomly chosen indices
#         idx120 = sorted(random.sample(list(range(1000)), k=120))
        idx120 = [16, 23, 52, 57, 76, 93, 95, 96, 99, 121, 122, 128, 148, 172, 181, 189, 202, 210, 232, 238, 257, 258, 259, 277, 283, 289, 295, 304, 307, 318, 322, 331, 337, 338, 345, 350, 361, 375, 376, 381, 388, 399, 401, 408, 424, 431, 432, 440, 447, 462, 464, 472, 483, 497, 506, 512, 530, 541, 553, 554, 557, 564, 570, 584, 612, 614, 619, 626, 631, 632, 650, 657, 658, 660, 674, 675, 680, 682, 691, 695, 699, 711, 734, 736, 741, 754, 757, 764, 769, 770, 780, 781, 787, 797, 799, 811, 822, 829, 830, 835, 837, 842, 843, 845, 873, 883, 897, 900, 902, 905, 913, 920, 925, 937, 938, 940, 941, 944, 949, 959]
        total_trainset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), transform=transform_train)
        testset = ImageNet(root=os.path.join(dataroot, 'imagenet-pytorch'), split='val', transform=transform_test)

        # compatibility
        total_trainset.targets = [lb for _, lb in total_trainset.samples]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=len(total_trainset) - 50000, random_state=0)  # 4000 trainset
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        train_idx, valid_idx = next(sss)

        # filter out
        train_idx = list(filter(lambda x: total_trainset.labels[x] in idx120, train_idx))
        valid_idx = list(filter(lambda x: total_trainset.labels[x] in idx120, valid_idx))
        test_idx = list(filter(lambda x: testset.samples[x][1] in idx120, range(len(testset))))

        targets = [idx120.index(total_trainset.targets[idx]) for idx in train_idx]
        for idx in range(len(total_trainset.samples)):
            if total_trainset.samples[idx][1] not in idx120:
                continue
            total_trainset.samples[idx] = (total_trainset.samples[idx][0], idx120.index(total_trainset.samples[idx][1]))
        total_trainset = Subset(total_trainset, train_idx)
        total_trainset.targets = targets

        for idx in range(len(testset.samples)):
            if testset.samples[idx][1] not in idx120:
                continue
            testset.samples[idx] = (testset.samples[idx][0], idx120.index(testset.samples[idx][1]))
        testset = Subset(testset, test_idx)
        print('reduced_imagenet train=', len(total_trainset))
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    if total_aug is not None and augs is not None:
        total_trainset.set_preaug(augs, total_aug)
        print('set_preaug-')

    train_sampler = None
    
    '''if multinode:
            train_sampler = torch.utils.data.distributed.DistributedSampler(total_trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            logger.info(f'----- dataset with DistributedSampler  {dist.get_rank()}/{dist.get_world_size()}')'''

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=8, pin_memory=False,
        sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_valset, batch_size=batch, shuffle=False, num_workers=4, pin_memory=False, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        total_testset, batch_size=batch, shuffle=False, num_workers=8, pin_memory=False,
        drop_last=False
    )
    return train_sampler, trainloader, validloader, testloader





def num_class(dataset):
    return {
        'cifar10': 10,
        'camelyon17': 2,
        'reduced_cifar10': 10,
        'cifar10.1': 10,
        'cifar100': 100,
        'reduced_cifar100': 100,
        'svhn': 10,
        'reduced_svhn': 10,
        'imagenet': 1000,
        'reduced_imagenet': 120,
    }[dataset]




class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
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


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                try:

                    img = apply_augment(img, name, level)
                    #img = apply_augment(Image.fromarray(np.uint8(img)), name, level)
                except Exception as e:
                    print("Exception raised: ",e)
                    print("data type: ",img.type())
        return img


class EfficientNetRandomCrop:
    def __init__(self, imgsize, min_covered=0.1, aspect_ratio_range=(3./4, 4./3), area_range=(0.08, 1.0), max_attempts=10):
        assert 0.0 < min_covered
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
        assert 0 < area_range[0] <= area_range[1]
        assert 1 <= max_attempts

        self.min_covered = min_covered
        self.aspect_ratio_range = aspect_ratio_range
        self.area_range = area_range
        self.max_attempts = max_attempts
        self._fallback = EfficientNetCenterCrop(imgsize)

    def __call__(self, img):
        # https://github.com/tensorflow/tensorflow/blob/9274bcebb31322370139467039034f8ff852b004/tensorflow/core/kernels/sample_distorted_bounding_box_op.cc#L111
        original_width, original_height = img.size
        min_area = self.area_range[0] * (original_width * original_height)
        max_area = self.area_range[1] * (original_width * original_height)

        for _ in range(self.max_attempts):
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            height = int(round(math.sqrt(min_area / aspect_ratio)))
            max_height = int(round(math.sqrt(max_area / aspect_ratio)))

            if max_height * aspect_ratio > original_width:
                max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
                max_height = int(max_height)
                if max_height * aspect_ratio > original_width:
                    max_height -= 1

            if max_height > original_height:
                max_height = original_height

            if height >= max_height:
                height = max_height

            height = int(round(random.uniform(height, max_height)))
            width = int(round(height * aspect_ratio))
            area = width * height

            if area < min_area or area > max_area:
                continue
            if width > original_width or height > original_height:
                continue
            if area < self.min_covered * (original_width * original_height):
                continue
            if width == original_width and height == original_height:
                return self._fallback(img)      # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L102

            x = random.randint(0, original_width - width)
            y = random.randint(0, original_height - height)
            return img.crop((x, y, x + width, y + height))

        return self._fallback(img)


class EfficientNetCenterCrop:
    def __init__(self, imgsize):
        self.imgsize = imgsize

    def __call__(self, img):
        """Crop the given PIL Image and resize it to desired size.
        Args:
            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            PIL Image: Cropped image.
        """
        image_width, image_height = img.size
        image_short = min(image_width, image_height)

        crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

        crop_height, crop_width = crop_size, crop_size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
