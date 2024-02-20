from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
"""
This repository is build upon RandAugment implementation
https://arxiv.org/abs/1909.13719 published here
https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
"""
#Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AutoAugment and RandAugment policies for enhanced image preprocessing.
AutoAugment Reference: https://arxiv.org/abs/1805.09501
RandAugment Reference: https://arxiv.org/abs/1909.13719
"""
import inspect
import numpy as np
import math
#from tensorflow.contrib import training as contrib_training
from PIL import Image, ImageEnhance, ImageOps, ImageFilter 
from augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from augmenters.color.hedcoloraugmenter import HedColorAugmenter
from augmenters.noise.gaussianbluraugmenter import GaussianBlurAugmenter
from augmenters.noise.additiveguassiannoiseaugmenter import AdditiveGaussianNoiseAugmenter
from albumentations.augmentations.geometric.functional import elastic_transform
from albumentations.augmentations.geometric.resize import RandomScale
from augmenters.spatial.scalingaugmenter import ScalingAugmenter
import random

# augmentation scheme.
_MAX_LEVEL = 10.
_REPLACE = 128

def scaling(image,factor):
    
    
    if random.random() > 0.5:
        factor = factor/60
        augmentor = ScalingAugmenter(scaling_range=(1-factor,3), interpolation_order=1)
    else:
        factor = factor/30
        augmentor = ScalingAugmenter(scaling_range=(1+factor,3), interpolation_order=1)
    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    
    
    #transform=RandomScale(scale_limit=factor, interpolation=1, always_apply=False, p=1)


    #print('_hsv_h',np.max(image))
    #image = transform.apply(img=image)
    return image#['image']


def hsv_h(image, factor):
    #image=PIL.Image.fromarray(image)
    #print('image',image.shape)

    #factor = random.uniform(0, factor)
    factor=factor/30

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


def hsv_s(image, factor):
    #factor = random.uniform(0, factor)
    #image=np.asarray(image)
    factor=factor/30
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

def hsv_v(image, factor):
    #factor = random.uniform(0, factor)
    #image=np.asarray(image)
    factor=factor/30
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


def hsv(image, factor):
    #factor = random.uniform(0, factor)
    #image=np.asarray(image)
    factor=factor/30
    if random.random() > 0.5:
        factor = -factor
    
    #print('image',image.shape)
    #image=np.transpose(image,[2,0,1])
    #print('image',image.shape)
    #print('hsv v factor',factor)
    augmentor= HsbColorAugmenter(hue_sigma_range=factor, saturation_sigma_range=factor, brightness_sigma_range=factor)
    #Not randomizing the augmentation magnitude 
    augmentor.randomize()
    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    '''
    if num < 0.01:
        image.save('/mnt/netcache/pathology/projects/autoaugmentation/data/saved_fastauto/'+'_hsv_v'+str(num)+'.jpg')

    '''
    #print('_hsv_v',np.max(image))
    return image#np.transpose(augmentor.transform(image),[1,2,0])

def hed(image, factor):
    #factor = random.uniform(0, factor)
    #image=np.asarray(image)
    factor=factor/30
    if random.random() > 0.5:
        factor = -factor
    #print('applying hed_h')
    #image=np.transpose(image,[2,0,1])
    #print('imagin hed_h imagee',image.shape)
    #image=np.transpose(image,[2,0,1])
    #print('hed h factor',factor)
    augmentor= HedColorAugmenter(haematoxylin_sigma_range=factor, haematoxylin_bias_range=factor,
                                            eosin_sigma_range=factor, eosin_bias_range=factor,
                                            dab_sigma_range=factor, dab_bias_range=factor,
                                            cutoff_range=(0.15, 0.85))
    #Not randomizing the augmentation magnitude 
    augmentor.randomize()
    image = augmentor.transform(image)
    #image = PIL.Image.fromarray(image)
    '''
    if num < 0.01:
        image.save('/mnt/netcache/pathology/projects/autoaugmentation/data/saved_fastauto/'+'_hed_h'+str(num)+'.jpg')

    '''
    #print('_hed_h',np.max(image))
    return image

def hed_h(image, factor):
    #factor = random.uniform(0, factor)
    #image=np.asarray(image)
    factor=factor/30
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

def hed_e(image, factor):
    #factor = random.uniform(0, factor)
    factor=factor/30
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

def hed_d(image, factor):
    #factor = random.uniform(0, factor)
    factor=factor/30
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
def gauss_blur(image, factor):
  """Equivalent of PIL Gaussian Blur."""
  factor=factor/5
  #image=np.transpose(image,[2,0,1])
  augmentor= GaussianBlurAugmenter(sigma_range=(factor, factor*10))
  #Not randomizing the augmentation magnitude 
  #augmentor.randomize()
  return augmentor.transform(image)#np.transpose(augmentor.transform(image),[1,2,0])

def gauss_noise(image, factor):
  """Equivalent of PIL Gaussian noise."""
  factor=factor/2
  
  augmentor= AdditiveGaussianNoiseAugmenter(sigma_range=(0.1*factor,factor))
   
  return augmentor.transform(image)#(augmentor.transform(image),[1,2,0])

def elastic(image, factor):
  """Equivalent of PIL Gaussian noise."""
  image=elastic_transform(image,alpha=factor*7, sigma=factor*7, alpha_affine=factor*7)
  return image

def color(image, factor):
  """Equivalent of PIL Color."""
  factor=factor/5+1
  image = Image.fromarray(image)
  image = ImageEnhance.Color(image).enhance(factor) 
  return np.asarray(image)


def contrast(image, factor):
  """Equivalent of PIL Contrast."""
  factor=factor/5+1
  image = Image.fromarray(image)
  image = ImageEnhance.Contrast(image).enhance(factor)
  return np.asarray(image)


def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  factor=factor/10+1
  image = Image.fromarray(image)
  image = ImageEnhance.Brightness(image).enhance(factor)
  return np.asarray(image)



def rotate(image,degrees, replace=(_REPLACE,_REPLACE,_REPLACE)):
    """Equivalent of PIL Posterize."""
    if random.random() > 0.5:
        degrees = -degrees
    image = Image.fromarray(image)
    image =  image.rotate(angle=degrees*10,fillcolor =replace)
    return np.asarray(image)




def translate_x(image, pixels, replace=(_REPLACE,_REPLACE,_REPLACE)):

    """Equivalent of PIL Translate in X dimension."""
    if random.random() > 0.5:
        pixels = -pixels
    pixels=pixels*3
    image = Image.fromarray(image)
    image=image.transform(image.size, Image.AFFINE, (1, 0,pixels, 0, 1, 0), fillcolor =replace)
    return np.asarray(image)

def translate_y(image, pixels, replace=(_REPLACE,_REPLACE,_REPLACE)):
  """Equivalent of PIL Translate in Y dimension."""
  if random.random() > 0.5:
        pixels = -pixels
  pixels=pixels*3
  image = Image.fromarray(image)
  image=image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),fillcolor =replace)
  return np.asarray(image)

def shear_x(image, level, replace=(_REPLACE,_REPLACE,_REPLACE)):
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  if random.random() > 0.5:
        level = -level
  level=level/20
  image = Image.fromarray(image)
  image=image.transform(image.size, Image.AFFINE, (1, level, 0, 0, 1, 0), Image.BICUBIC, fillcolor =replace)
  return np.asarray(image)


def shear_y(image, level, replace=(_REPLACE,_REPLACE,_REPLACE)):
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  level=level/20
  if random.random() > 0.5:
        level = -level
  image = Image.fromarray(image)
  image=image.transform(image.size, Image.AFFINE, (1, 0, 0,level,  1, 0), Image.BICUBIC, fillcolor =replace)
  return np.asarray(image)


def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops.
  Args:
    image: A 3D uint8 tensor.
  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """
  image = Image.fromarray(image)
  image =  ImageOps.autocontrast(image)
  return np.asarray(image)


def identity(image):
  """Implements Identity
 
  """
  return image
  
def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  image = Image.fromarray(image)
  image =  ImageEnhance.Sharpness(image).enhance(factor)
  return np.asarray(image)



def equalize(image):
  """Implements Equalize function from PIL using TF ops."""
  image = Image.fromarray(image)
  image =  ImageOps.equalize(image) 
  return np.asarray(image)
 
'''
    'HsvH': hsv_h,
    'HsvS': hsv_s,
    'HsvV': hsv_v,
    'HedH': hed_h,
    'HedE': hed_e,
    'HedD': hed_d,
'''



NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'HsvH': hsv_h,
    'HsvS': hsv_s,
    'HsvV': hsv_v,
    'HedH': hed_h,
    'HedE': hed_e,
    'HedD': hed_d,
    'Hsv': hsv,
    'Hed': hed,
    'Identity': identity,
    'Equalize': equalize,
    'Rotate': rotate,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
    'Elastic': elastic,
    'GaussBlur': gauss_blur,
    'GaussNoise': gauss_noise,
    'Scaling': scaling


}


def _randomly_negate_tensor(tensor):
  """With 50% prob turn the tensor negative."""
  rand_cva = list([1, 0])
  
  should_flip = random.choice(rand_cva)
  
  if should_flip == 1:
      final_tensor = tensor
  else:  
      final_tensor = -tensor
  return final_tensor




def _rotate_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 30.
  level = _randomly_negate_tensor(level)
  return (level,)


def _shrink_level_to_arg(level):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (_MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)
  
def _enhance_level_to_arg_hsv(level):
  return (level*0.03,)
  
def _enhance_level_to_arg_hed(level):
  return (level*0.03,)
  
def _enhance_level_to_arg_contrast(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)
  
def _enhance_level_to_arg_brightness(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)
  
def _enhance_level_to_arg_color(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)



def _shear_level_to_arg(level):
  level = (level/_MAX_LEVEL) * 0.3
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)

def _level_to_arg(level):

  return (level,)

def _translate_level_to_arg(level, translate_const):
  level = (level/_MAX_LEVEL) * float(translate_const)
  # Flip level to negative with 50% chance.
  level = _randomly_negate_tensor(level)
  return (level,)

'''
      'HsvH': _level_to_arg,
      'HsvS': _level_to_arg,
      'HsvV': _level_to_arg,
      'HedH': _level_to_arg,
      'HedE': _level_to_arg,
      'HedD': _level_to_arg,'''



def level_to_arg(hparams):
  return {
      'Identity': lambda level: (),
      'Hsv': _level_to_arg,
      'Hed': _level_to_arg,
      'HsvH': _level_to_arg,
      'HsvS': _level_to_arg,
      'HsvV': _level_to_arg,
      'HedH': _level_to_arg,
      'HedE': _level_to_arg,
      'HedD': _level_to_arg,
      'AutoContrast': lambda level: (),
      'Equalize': lambda level: (),
      'Rotate': _level_to_arg,
      'Color': _level_to_arg,
      'Contrast': _level_to_arg,
      'Brightness': _level_to_arg,
      'Sharpness': _level_to_arg,
      'ShearX': _level_to_arg,
      'ShearY': _level_to_arg,
      'TranslateX': _level_to_arg,
      'TranslateY': _level_to_arg,
      'Elastic': _level_to_arg,
      'GaussBlur': _level_to_arg,
      'GaussNoise': _level_to_arg,
      'Scaling': _level_to_arg,
  }


def _parse_policy_info(name, prob, level, replace_value, augmentation_hparams,magnitude):
  """Return the function that corresponds to `name` and update `level` param."""

  func = NAME_TO_FUNC[name]
  args = level_to_arg(augmentation_hparams)[name](level)
  if name == 'Hed':
    args = level_to_arg(augmentation_hparams)[name](magnitude)
  elif name == 'Hsv':
    args = level_to_arg(augmentation_hparams)[name](magnitude)


  # Check to see if prob is passed into function. This is used for operations
  # where we alter bboxes independently.
  # pytype:disable=wrong-arg-types
  if 'prob' in inspect.getargspec(func)[0]:
    args = tuple([prob] + list(args))
  # pytype:enable=wrong-arg-types

  # Add in replace arg if it is required for the function that is being called.
  # pytype:disable=wrong-arg-types
  #if 'replace' in inspect.getargspec(func)[0]:
  #  # Make sure replace is the final argument
  #  assert 'replace' == inspect.getargspec(func)[0][-1]
  #  args = tuple(list(args) + [replace_value])
  # pytype:enable=wrong-arg-types

  return (func, prob, args)


def _apply_func_with_prob(func, image, args, prob):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)

  # If prob is a function argument, then this randomness is being handled
  # inside the function, so make sure it is always called.
  # pytype:disable=wrong-arg-types
  if 'prob' in inspect.getargspec(func)[0]:
    prob = 1.0
  # pytype:enable=wrong-arg-types

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random_uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image = tf.cond(
      should_apply_op,
      lambda: func(image, *args),
      lambda: image)
  return augmented_image


def select_and_apply_random_policy(policies, image):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random_uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image),
        lambda: image)
  return image





def distort_image_with_randaugment(image, num_layers, magnitude, randomize=True,randaugment_transforms_set='review'):
  """Applies the RandAugment policy to `image`.
  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  Args:
    image: `Tensor` of shape [height, width, 3] representing an image.
    num_layers: Integer, the number of augmentation transformations to apply
      sequentially to an image. Represented as (N) in the paper. Usually best
      values will be in the range [1, 3].
    magnitude: Integer, shared magnitude across all augmentation operations.
      Represented as (M) in the paper. Usually best values are in the range
      [1, 10].
  Returns:
    The augmented version of `image`.
  """
  #print(magnitude)
  replace_value = (128, 128, 128) #[128] * 3
  #tf.logging.info('Using RandAug.')
  augmentation_hparams = None #contrib_training.HParams(cutout_const=40, translate_const=10)
  #print('augmentation_hparams',augmentation_hparams)
  #The 'Default' option is the H&E tailored randaugment
  if randaugment_transforms_set=='review':
      available_ops = ['Scaling','TranslateX', 'TranslateY','ShearX', 'ShearY','Brightness', 'Sharpness','Color', 'Contrast','Rotate', 'Equalize','Identity','HsvH','HsvS','HsvV','HedH','HedE','HedD', 'Elastic','GaussBlur','GaussNoise']  
  elif randaugment_transforms_set=='midl':
      available_ops = ['Scaling','TranslateX', 'TranslateY','ShearX', 'ShearY','Brightness', 'Sharpness','Color', 'Contrast','Rotate', 'Equalize','Identity','Hsv','Hed', 'Elastic','GaussBlur','GaussNoise']  
  

  #available_ops = ['TranslateX', 'TranslateY','ShearX', 'ShearY','Brightness', 'Sharpness','Color', 'Contrast','Rotate', 'Identity','Hsv','Hed']  

  for layer_num in range(num_layers):
    op_to_select = np.random.randint(low=0,high=len(available_ops))
    if randomize:
      random_magnitude = np.random.uniform(low=0, high=magnitude)
    else:
      random_magnitude = magnitude
    
    for (i, op_name) in enumerate(available_ops):
        prob = np.random.uniform(low=0.2, high=0.8)

        func, _, args = _parse_policy_info(op_name, prob, random_magnitude,
                                           replace_value, augmentation_hparams,magnitude)

        if  (i== op_to_select):

            selected_func=func
            selected_args=args
            image= selected_func(image, *selected_args)
        else: 
            image=image
  return image