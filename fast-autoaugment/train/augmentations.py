# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw, PIL.Image
import numpy as np
import torch
from torchvision.transforms.transforms import Compose
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
from augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from augmenters.color.hedcoloraugmenter import HedColorAugmenter
from augmenters.noise.gaussianbluraugmenter import GaussianBlurAugmenter
from augmenters.noise.additiveguassiannoiseaugmenter import AdditiveGaussianNoiseAugmenter
from albumentations.augmentations.geometric.functional import elastic_transform
from augmenters.spatial.scalingaugmenter import ScalingAugmenter
random_mirror = True

def _gauss_blur(image, factor):
  """Equivalent of PIL Gaussian Blur."""
  #image=np.transpose(image,[2,0,1])
  augmentor= GaussianBlurAugmenter(sigma_range=(0.1*factor, factor))
  #Not randomizing the augmentation magnitude 
  #augmentor.randomize()
  return augmentor.transform(image)#np.transpose(augmentor.transform(image),[1,2,0])

def _gauss_noise(image, factor):
  """Equivalent of PIL Gaussian noise."""
  
  augmentor= AdditiveGaussianNoiseAugmenter(sigma_range=(0.1*factor,factor))
   
  return augmentor.transform(image)#(augmentor.transform(image),[1,2,0])

def _elastic(image, factor):
  """Equivalent of PIL Gaussian noise."""
  image=elastic_transform(image,alpha=factor, sigma=factor, alpha_affine=factor)
  return image

def _scaling(image,factor):
    augmentor = ScalingAugmenter(scaling_range=(1-factor, 1+factor), interpolation_order=1)
    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    '''
    if num < 0.01:
        image.save('/mnt/netcache/pathology/projects/autoaugmentation/data/saved_fastauto/'+'_hsv_h'+str(num)+'.jpg')

    '''
    #print('_hsv_h',np.max(image))
    return image

def _hsv_h(image, factor):
    #image=PIL.Image.fromarray(image)
    #print('image',image.shape)

    #factor = random.uniform(0, factor)
    if random.random() > 0.5:
        factor = -factor
    #image=np.asarray(image)
    #print('hsv h factor',factor)
    augmentor= HsbColorAugmenter(hue_sigma_range = factor, saturation_sigma_range=0, brightness_sigma_range=0)
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()

    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    '''
    if num < 0.01:
        image.save('/mnt/netcache/pathology/projects/autoaugmentation/data/saved_fastauto/'+'_hsv_h'+str(num)+'.jpg')

    '''
    #print('_hsv_h',np.max(image))
    return image#np.transpose(augmentor.transform(image),[1,2,0])
    
def _hsv_s(image, factor):
    #factor = random.uniform(0, factor)
    #image=np.asarray(image)
    if random.random() > 0.5:
        factor = -factor
    #print('image',image.shape)
    #augmentor.transform(image)#
    #print('hsv s factor',factor)
    augmentor= HsbColorAugmenter(hue_sigma_range=0, saturation_sigma_range=factor, brightness_sigma_range=0)
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    '''
    if num < 0.01:
        image.save('/mnt/netcache/pathology/projects/autoaugmentation/data/saved_fastauto/'+'_hsv_s'+str(num)+'.jpg')

    '''
    #print('_hsv_s',np.max(image))
    return image#np.transpose(augmentor.transform(image),[1,2,0])  

def _hsv_v(image, factor):
    #factor = random.uniform(0, factor)
    #image=np.asarray(image)
    if random.random() > 0.5:
        factor = -factor
    
    #print('image',image.shape)
    #image=np.transpose(image,[2,0,1])
    #print('image',image.shape)
    #print('hsv v factor',factor)
    augmentor= HsbColorAugmenter(hue_sigma_range=0, saturation_sigma_range=0, brightness_sigma_range=factor)
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    '''
    if num < 0.01:
        image.save('/mnt/netcache/pathology/projects/autoaugmentation/data/saved_fastauto/'+'_hsv_v'+str(num)+'.jpg')

    '''
    #print('_hsv_v',np.max(image))
    return image#np.transpose(augmentor.transform(image),[1,2,0])    
    
def _hed_h(image, factor):
    #factor = random.uniform(0, factor)
    #image=np.asarray(image)
    if random.random() > 0.5:
        factor = -factor
    #print('applying hed_h')
    #image=np.transpose(image,[2,0,1])
    #print('imagin hed_h imagee',image.shape)
    #image=np.transpose(image,[2,0,1])
    #print('hed h factor',factor)
    augmentor= HedColorAugmenter(haematoxylin_sigma_range=factor, haematoxylin_bias_range=factor,
                                            eosin_sigma_range=0, eosin_bias_range=0,
                                            dab_sigma_range=0, dab_bias_range=0,
                                            cutoff_range=(0.15, 0.85))
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    '''
    if num < 0.01:
        image.save('/mnt/netcache/pathology/projects/autoaugmentation/data/saved_fastauto/'+'_hed_h'+str(num)+'.jpg')

    '''
    #print('_hed_h',np.max(image))
    return image#np.transpose(augmentor.transform(image),[1,2,0])

def _hed_e(image, factor):
    #factor = random.uniform(0, factor)
    if random.random() > 0.5:
        factor = -factor
    #print('image',image.shape)
    #image=np.transpose(image,[2,0,1])
    #image=np.asarray(image)
    #print('in hed_e image',image.shape)
    #print('hed e factor',factor)
    augmentor= HedColorAugmenter(haematoxylin_sigma_range=0, haematoxylin_bias_range=0,
                                            eosin_sigma_range=factor, eosin_bias_range=factor,
                                            dab_sigma_range=0, dab_bias_range=0,
                                            cutoff_range=(0.15, 0.85))
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    '''
    if num < 0.01:
        image.save('/mnt/netcache/pathology/projects/autoaugmentation/data/saved_fastauto/'+'_hed_e'+str(num)+'.jpg')

    '''
    #print('_hed_e',np.max(image))
    return image#np.transpose(augmentor.transform(image),[1,2,0])

def _hed_d(image, factor):
    #factor = random.uniform(0, factor)
    if random.random() > 0.5:
        factor = -factor
    #print('image',image.shape)
    #image=np.transpose(image,[2,0,1])
    #image=np.asarray(image)
    #print('in hed_e image',image.shape)
    #print('hed e factor',factor)
    augmentor= HedColorAugmenter(haematoxylin_sigma_range=0, haematoxylin_bias_range=0,
                                            eosin_sigma_range=0, eosin_bias_range=0,
                                            dab_sigma_range=factor, dab_bias_range=factor,
                                            cutoff_range=(0.15, 0.85))
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    '''
    if num < 0.01:
        image.save('/mnt/netcache/pathology/projects/autoaugmentation/data/saved_fastauto/'+'_hed_e'+str(num)+'.jpg')

    '''
    #print('_hed_e',np.max(image))
    return image#np.transpose(augmentor.transform(image),[1,2,0])



def ShearX(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3

    v = random.uniform(0, v)
    img = PIL.Image.fromarray(img)
    if random_mirror and random.random() > 0.5:
        v = -v
    img=np.asarray(img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0)))
    #print('ShearX',np.max(img))

    return img

def ShearY(img, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    v = random.uniform(0, v)
    img = PIL.Image.fromarray(img)
    if random_mirror and random.random() > 0.5:
        v = -v
    img = np.asarray(img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0)))
    #print('ShearY',np.max(img))

    return img


def TranslateX(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]

    assert -0.45 <= v <= 0.45
    v = random.uniform(0, v)
    img = PIL.Image.fromarray(img)
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    img =  np.asarray(img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)))
    #print('TranslateX',np.max(img))

    return img


def TranslateY(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    v = random.uniform(0, v)
    img = PIL.Image.fromarray(img)
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    img = np.asarray(img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)))
    #print('TranslateY',np.max(img))

    return img


def TranslateXAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    v = random.uniform(0, v)
    img = PIL.Image.fromarray(img)
    if random.random() > 0.5:
        v = -v
    return np.asarray(img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0)))


def TranslateYAbs(img, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v <= 10
    v = random.uniform(0, v)
    img = PIL.Image.fromarray(img)
    if random.random() > 0.5:
        v = -v
    return np.asarray(img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v)))


def Rotate(img, v):  # [-30, 30]
    assert -30 <= v <= 30
    v = random.uniform(0, v)
    img = PIL.Image.fromarray(img)
    if random_mirror and random.random() > 0.5:
        v = -v
    return np.asarray(img.rotate(v))


def AutoContrast(img, _):
    img = PIL.Image.fromarray(img)
    return np.asarray(PIL.ImageOps.autocontrast(img))


def Invert(img, _):
    img = PIL.Image.fromarray(img)
    return np.asarray(PIL.ImageOps.invert(img))


def Equalize(img, _):
    img = PIL.Image.fromarray(img)
    return np.asarray(PIL.ImageOps.equalize(img))


def Flip(img, _):  # not from the paper
    img = PIL.Image.fromarray(img)
    return np.asarray(PIL.ImageOps.mirror(img))


def Solarize(img, v):  # [0, 256]
    img = PIL.Image.fromarray(img)
    return np.asarray(PIL.ImageOps.solarize(img, v))


def Posterize(img, v):  # [4, 8]
    assert 4 <= v <= 8
    img = PIL.Image.fromarray(img)
    v = int(v)
    return np.asarray(PIL.ImageOps.posterize(img, v))


def Posterize2(img, v):  # [0, 4]
    assert 0 <= v <= 4
    #img = PIL.Image.fromarray(img)
    v = int(v)
    return np.asarray(PIL.ImageOps.posterize(img, v))


def Contrast(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    v = random.uniform(0.1, v)
    img = PIL.Image.fromarray(img)
    return np.asarray(PIL.ImageEnhance.Contrast(img).enhance(v))


def Color(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    v = random.uniform(0.1, v)
    img = PIL.Image.fromarray(img)
    return np.asarray(PIL.ImageEnhance.Color(img).enhance(v))


def Brightness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    v = random.uniform(0.1, v)
    img = PIL.Image.fromarray(img)
    return np.asarray(PIL.ImageEnhance.Brightness(img).enhance(v))


def Sharpness(img, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    v = random.uniform(0.1, v)
    img = PIL.Image.fromarray(img)
    return np.asarray(PIL.ImageEnhance.Sharpness(img).enhance(v))


def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    img = PIL.Image.fromarray(img)
    if v <= 0.:
        return img

    v = v * img.size[0]
    return np.asarray(CutoutAbs(img, v))


def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    img = PIL.Image.fromarray(img)
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    color = (125, 123, 114)
    # color = (0, 0, 0)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return np.asarray(img)


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return np.asarray(f)


def augment_list(for_autoaug=False):  # 16 oeprations and their ranges
    l = [
        (ShearX, -0.3, 0.3),  # 0
        (ShearY, -0.3, 0.3),  # 1
        (TranslateX, -0.45, 0.45),  # 2
        (TranslateY, -0.45, 0.45),  # 3
        (Rotate, -30, 30),  # 4
        (AutoContrast, 0, 1),  # 5
        (_hsv_h, 0, 1.0),  # 6
        (Equalize, 0, 1),  # 7
        (_hsv_s, 0.0, 1.0),  # 8
        (_hsv_v, 0.0, 1.0),  # 9
        (Contrast, 0.1, 1.9),  # 10
        (Color, 0.1, 1.9),  # 11
        (Brightness, 0.1, 1.9),  # 12
        (Sharpness, 0.1, 1.9),  # 13
        (_hed_h, 0.0, 1.0),  # 14
        (_hed_e, 0.0, 1.0),  # 15
        (_hed_d, 0.0, 1.0),  # 16
        (_gauss_blur, 0.0, 30.0),  # 17
        (_gauss_noise, 0.0, 30.0),  # 18
        (_scaling, 0.0, 0.5) , # 19
        (_elastic, 0.0, 300.0) # 20
    ]
    
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    #print('func',augment_fn)
    #print('img MAX',np.max(img))
    #print('Object type:',type(img))
    return augment_fn(img.copy(), level * (high - low) + low)


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
