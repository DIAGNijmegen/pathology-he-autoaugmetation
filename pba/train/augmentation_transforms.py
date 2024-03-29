# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""Transforms used in the Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import PIL
import inspect
import random
import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image

MEANS = {
    'cifar10_50000': [0.49139968, 0.48215841, 0.44653091],
    'cifar10_4000': [0.49056774, 0.48116026, 0.44726052],
    'cifar100_50000': [0.50707516, 0.48654887, 0.44091784],
    'svhn_1000': [0.45163885, 0.4557915, 0.48093327],
    'svhn-full_604388': [0.43090966, 0.4302428, 0.44634357],
    'camelyon17_5400': [0.49139968, 0.48215841, 0.44653091],
    'camelyon17_3600': [0.49139968, 0.48215841, 0.44653091]
}
STDS = {
    'cifar10_50000': [0.24703223, 0.24348513, 0.26158784],
    'cifar10_4000': [0.24710728, 0.24451308, 0.26235099],
    'cifar100_50000': [0.26733429, 0.25643846, 0.27615047],
    'svhn_1000': [0.20385217, 0.20957996, 0.20804394],
    'svhn-full_604388': [0.19652855, 0.19832038, 0.19942076],
    'camelyon17_5400': [0.19652855, 0.19832038, 0.19942076],
    'camelyon17_3600': [0.19652855, 0.19832038, 0.19942076]
}
PARAMETER_MAX = 10  # What is the max 'level' a transform could be predicted


from augmenters.color.hsbcoloraugmenter import HsbColorAugmenter
from augmenters.color.hedcoloraugmenter import HedColorAugmenter
from augmenters.noise.gaussianbluraugmenter import GaussianBlurAugmenter
from augmenters.noise.additiveguassiannoiseaugmenter import AdditiveGaussianNoiseAugmenter
from albumentations_elastic.augmentations.geometric.functional import elastic_transform
from augmenters.spatial.scalingaugmenter import ScalingAugmenter
random_mirror = True


def _gauss_blur(image, factor):
  """Equivalent of PIL Gaussian Blur."""
  image = np.asarray(image)
  #image=np.transpose(image,[2,0,1])
  factor = float_parameter(factor, 6)
  augmentor= GaussianBlurAugmenter(sigma_range=(factor, 10*factor))
  #Not randomizing the augmentation magnitude 
  #augmentor.randomize()
  return PIL.Image.fromarray(augmentor.transform(image))#np.transpose(augmentor.transform(image),[1,2,0])

def _gauss_noise(image, factor):
  """Equivalent of PIL Gaussian noise."""
  image = np.asarray(image)
  factor = float_parameter(factor, 2)
  
  augmentor= AdditiveGaussianNoiseAugmenter(sigma_range=(factor,10*factor))
   
  return PIL.Image.fromarray(augmentor.transform(image))#(augmentor.transform(image),[1,2,0])

def _elastic(image, factor):
  """Equivalent of PIL Gaussian noise."""
  image = np.asarray(image)
  factor = int_parameter(factor, 210)
  image=elastic_transform(image,alpha=factor, sigma=factor, alpha_affine=factor)
  return PIL.Image.fromarray(image)

def _scaling(image,factor):
    
    image = np.asarray(image)
    if random.random() > 0.5:
        factor = float_parameter(factor, 0.5)
        augmentor = ScalingAugmenter(scaling_range=(1-factor,3), interpolation_order=1)
    else:
        factor = float_parameter(factor, 1.0)
        augmentor = ScalingAugmenter(scaling_range=(1+factor,3), interpolation_order=1)
    image = augmentor.transform(image)

    return PIL.Image.fromarray(image)

def _hsv_h(image, factor):
    #image=PIL.Image.fromarray(image)
    #print('image',image.shape)
    #factor = random.uniform(0, factor)
    if random.random() > 0.5:
        factor = -factor
    image=np.asarray(image)
    print('hsv_h max image value:',np.max(image))

    print('hsv_h factor value:',factor)
    #print('hsv h factor',factor)
    augmentor= HsbColorAugmenter(hue_sigma_range = factor, saturation_sigma_range=0, brightness_sigma_range=0)
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    return PIL.Image.fromarray(augmentor.transform(image))#np.transpose(augmentor.transform(image),[1,2,0])
    
def _hsv_s(image, factor):
    factor = float_parameter(factor, 1)
    #factor = random.uniform(0, factor)
    image=np.asarray(image)
    
    if random.random() > 0.5:
        factor = -factor
    print('hsv_s max image value:',np.max(image))

    print('hsv_s factor value:',factor)
    #print('image',image.shape)
    #augmentor.transform(image)#
    #print('hsv s factor',factor)
    augmentor= HsbColorAugmenter(hue_sigma_range=0, saturation_sigma_range=factor, brightness_sigma_range=0)
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    return PIL.Image.fromarray(augmentor.transform(image))#np.transpose(augmentor.transform(image),[1,2,0])  

def _hsv_v(image, factor):
    factor = float_parameter(factor, 1)
    #factor = random.uniform(0, factor)
    image=np.asarray(image)
    
    if random.random() > 0.5:
        factor = -factor
    print('hsv_v max image value:',np.max(image))

    print('hsv_v factor value:',factor)
    #print('image',image.shape)
    #image=np.transpose(image,[2,0,1])
    #print('image',image.shape)
    #print('hsv v factor',factor)
    augmentor= HsbColorAugmenter(hue_sigma_range=0, saturation_sigma_range=0, brightness_sigma_range=factor)
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    return PIL.Image.fromarray(augmentor.transform(image))#np.transpose(augmentor.transform(image),[1,2,0])    
    
def _hed_h(image, factor):
    factor = float_parameter(factor, 1)
    #factor = random.uniform(0, factor)
    image=np.asarray(image)
    
    if random.random() > 0.5:
        factor = -factor
    print('hed_h max image value:',np.max(image))

    print('hed_h factor value:',factor)
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
    return PIL.Image.fromarray(augmentor.transform(image))#np.transpose(augmentor.transform(image),[1,2,0])

def _hed_e(image, factor):
    factor = float_parameter(factor, 1)
    #factor = random.uniform(0, factor)
    if random.random() > 0.5:
        factor = -factor
    #print('image',image.shape)
    #image=np.transpose(image,[2,0,1])
    image=np.asarray(image)
    print('hed_e max image value:',np.max(image))

    print('hed_e factor value:',factor)
    #print('in hed_e image',image.shape)
    #print('hed e factor',factor)
    augmentor= HedColorAugmenter(haematoxylin_sigma_range=0, haematoxylin_bias_range=0,
                                            eosin_sigma_range=factor, eosin_bias_range=factor,
                                            dab_sigma_range=0, dab_bias_range=0,
                                            cutoff_range=(0.15, 0.85))
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    return PIL.Image.fromarray(augmentor.transform(image))#np.transpose(augmentor.transform(image),[1,2,0])

def _hed_d(image, factor):
    factor = float_parameter(factor, 1)
    #factor = random.uniform(0, factor)
    if random.random() > 0.5:
        factor = -factor
    #print('image',image.shape)
    #image=np.transpose(image,[2,0,1])
    image=np.asarray(image)
    print('hed_d max image value:',np.max(image))

    print('hed_d factor value:',factor)
    #print('in hed_e image',image.shape)
    #print('hed e factor',factor)
    augmentor= HedColorAugmenter(haematoxylin_sigma_range=0, haematoxylin_bias_range=0,
                                            eosin_sigma_range=0, eosin_bias_range=0,
                                            dab_sigma_range=factor, dab_bias_range=factor,
                                            cutoff_range=(0.15, 0.85))
    #Not randomizing the augmentation magnitude 
    #augmentor.randomize()
    return PIL.Image.fromarray(augmentor.transform(image))#np.transpose(augmentor.transform(image),[1,2,0])

def pil_wrap(img, dataset):
    """Convert the `img` numpy tensor to a PIL Image."""
    return Image.fromarray(
        np.uint8(
            (img * STDS[dataset] + MEANS[dataset]) * 255.0)).convert('RGBA')


def pil_unwrap(pil_img, dataset, image_size):
    """Converts the PIL img to a numpy array."""
    pic_array = (np.array(pil_img.getdata()).reshape((image_size, image_size, 4)) / 255.0)
    i1, i2 = np.where(pic_array[:, :, 3] == 0)
    pic_array = (pic_array[:, :, :3] - MEANS[dataset]) / STDS[dataset]
    pic_array[i1, i2] = [0, 0, 0]
    return pic_array


def apply_policy(policy, img, dset, image_size):
    """Apply the `policy` to the numpy `img`.

  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what strength
      the operation to apply.
    img: Numpy image that will have `policy` applied to it.
    dset: Dataset, one of the keys of MEANS or STDS.
    image_size: Width and height of image.

  Returns:
    The result of applying `policy` to `img`.
  """
    pil_img = pil_wrap(img, dset)
    for xform in policy:
        assert len(xform) == 3
        name, probability, level = xform
        xform_fn = NAME_TO_TRANSFORM[name].pil_transformer(
            probability, level, image_size)
        pil_img = xform_fn(pil_img)
    return pil_unwrap(pil_img, dset, image_size)


def random_flip(x):
    """Flip the input x horizontally with 50% probability."""
    if np.random.rand(1)[0] > 0.5:
        return np.fliplr(x)
    return x


def zero_pad_and_crop(img, amount=4):
    """Zero pad by `amount` zero pixels on each side then take a random crop.

  Args:
    img: numpy image that will be zero padded and cropped.
    amount: amount of zeros to pad `img` with horizontally and verically.

  Returns:
    The cropped zero padded img. The returned numpy array will be of the same
    shape as `img`.
  """
    padded_img = np.zeros((img.shape[0] + amount * 2,
                           img.shape[1] + amount * 2, img.shape[2]))
    padded_img[amount:img.shape[0] + amount, amount:img.shape[1] +
               amount, :] = img
    top = np.random.randint(low=0, high=2 * amount)
    left = np.random.randint(low=0, high=2 * amount)
    new_img = padded_img[top:top + img.shape[0], left:left + img.shape[1], :]
    return new_img


def create_cutout_mask(img_height, img_width, num_channels, size):
    """Creates a zero mask used for cutout of shape `img_height` x `img_width`.

  Args:
    img_height: Height of image cutout mask will be applied to.
    img_width: Width of image cutout mask will be applied to.
    num_channels: Number of channels in the image.
    size: Size of the zeros mask.

  Returns:
    A mask of shape `img_height` x `img_width` with all ones except for a
    square of zeros of shape `size` x `size`. This mask is meant to be
    elementwise multiplied with the original image. Additionally returns
    the `upper_coord` and `lower_coord` which specify where the cutout mask
    will be applied.
  """
    assert img_height == img_width

    # Sample center where cutout mask will be applied
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)

    # Determine upper right and lower left corners of patch
    upper_coord = (max(0, height_loc - size // 2), max(0,
                                                       width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2),
                   min(img_width, width_loc + size // 2))
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]
    assert mask_height > 0
    assert mask_width > 0

    mask = np.ones((img_height, img_width, num_channels))
    zeros = np.zeros((mask_height, mask_width, num_channels))
    mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = (
        zeros)
    return mask, upper_coord, lower_coord


def cutout_numpy(img, size=16):
    """Apply cutout with mask of shape `size` x `size` to `img`.

  The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
  This operation applies a `size`x`size` mask of zeros to a random location
  within `img`.

  Args:
    img: Numpy image that cutout will be applied to.
    size: Height/width of the cutout mask that will be

  Returns:
    A numpy tensor that is the result of applying the cutout mask to `img`.
  """
    img_height, img_width, num_channels = (img.shape[0], img.shape[1],
                                           img.shape[2])
    assert len(img.shape) == 3
    mask, _, _ = create_cutout_mask(img_height, img_width, num_channels, size)
    return img * mask


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
    return int(level * maxval / PARAMETER_MAX)


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, pil_img):
        return self.f(pil_img)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def pil_transformer(self, probability, level, image_size):
        def return_function(im):
            if random.random() < probability:
                if 'image_size' in inspect.getargspec(self.xform).args:
                    im = self.xform(im, level, image_size)
                else:
                    im = self.xform(im, level)
            return im

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)


################## Transform Functions ##################
identity = TransformT('identity', lambda pil_img, level: pil_img)
flip_lr = TransformT(
    'FlipLR', lambda pil_img, level: pil_img.transpose(Image.FLIP_LEFT_RIGHT))
flip_ud = TransformT(
    'FlipUD', lambda pil_img, level: pil_img.transpose(Image.FLIP_TOP_BOTTOM))
# pylint:disable=g-long-lambda
auto_contrast = TransformT(
    'AutoContrast',
    lambda pil_img, level: ImageOps.autocontrast(pil_img.convert('RGB')).convert('RGBA')
)
equalize = TransformT(
    'Equalize',
    lambda pil_img, level: ImageOps.equalize(pil_img.convert('RGB')).convert('RGBA')
)
invert = TransformT(
    'Invert',
    lambda pil_img, level: ImageOps.invert(pil_img.convert('RGB')).convert('RGBA')
)
# pylint:enable=g-long-lambda
blur = TransformT('Blur',
                  lambda pil_img, level: pil_img.filter(ImageFilter.BLUR))
smooth = TransformT('Smooth',
                    lambda pil_img, level: pil_img.filter(ImageFilter.SMOOTH))


def _rotate_impl(pil_img, level):
    """Rotates `pil_img` from -30 to 30 degrees depending on `level`."""
    degrees = int_parameter(level, 30)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees)


rotate = TransformT('Rotate', _rotate_impl)


def _posterize_impl(pil_img, level):
    """Applies PIL Posterize to `pil_img`."""
    level = int_parameter(level, 4)
    return ImageOps.posterize(pil_img.convert('RGB'),
                              4 - level).convert('RGBA')


posterize = TransformT('Posterize', _posterize_impl)








def _shear_x_impl(pil_img, level, image_size):
    """Applies PIL ShearX to `pil_img`.

  The ShearX operation shears the image along the horizontal axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((image_size, image_size), Image.AFFINE, (1, level, 0, 0, 1, 0))


shear_x = TransformT('ShearX', _shear_x_impl)


def _shear_y_impl(pil_img, level, image_size):
    """Applies PIL ShearY to `pil_img`.

  The ShearY operation shears the image along the vertical axis with `level`
  magnitude.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had ShearX applied to it.
  """
    level = float_parameter(level, 0.3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((image_size, image_size), Image.AFFINE, (1, 0, 0, level, 1, 0))


shear_y = TransformT('ShearY', _shear_y_impl)


def _translate_x_impl(pil_img, level, image_size):
    """Applies PIL TranslateX to `pil_img`.

  Translate the image in the horizontal direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateX applied to it.
  """
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((image_size, image_size), Image.AFFINE, (1, 0, level, 0, 1, 0))


translate_x = TransformT('TranslateX', _translate_x_impl)


def _translate_y_impl(pil_img, level, image_size):
    """Applies PIL TranslateY to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had TranslateY applied to it.
  """
    level = int_parameter(level, 10)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform((image_size, image_size), Image.AFFINE, (1, 0, 0, 0, 1, level))


translate_y = TransformT('TranslateY', _translate_y_impl)


def _crop_impl(pil_img, level, image_size, interpolation=Image.BILINEAR):
    """Applies a crop to `pil_img` with the size depending on the `level`."""
    cropped = pil_img.crop((level, level, image_size - level,
                            image_size - level))
    resized = cropped.resize((image_size, image_size), interpolation)
    return resized


crop_bilinear = TransformT('CropBilinear', _crop_impl)


def _solarize_impl(pil_img, level):
    """Applies PIL Solarize to `pil_img`.

  Translate the image in the vertical direction by `level`
  number of pixels.

  Args:
    pil_img: Image in PIL object.
    level: Strength of the operation specified as an Integer from
      [0, `PARAMETER_MAX`].

  Returns:
    A PIL Image that has had Solarize applied to it.
  """
    level = int_parameter(level, 256)
    return ImageOps.solarize(pil_img.convert('RGB'),
                             256 - level).convert('RGBA')


solarize = TransformT('Solarize', _solarize_impl)


def _cutout_pil_impl(pil_img, level, image_size):
    """Apply cutout to pil_img at the specified level."""
    size = int_parameter(level, 20)
    if size <= 0:
        return pil_img
    img_height, img_width, num_channels = (image_size, image_size, 3)
    _, upper_coord, lower_coord = (create_cutout_mask(img_height, img_width,
                                                      num_channels, size))
    pixels = pil_img.load()  # create the pixel map
    for i in range(upper_coord[0], lower_coord[0]):  # for every col:
        for j in range(upper_coord[1], lower_coord[1]):  # For every row
            pixels[i, j] = (125, 122, 113, 0)  # set the colour accordingly
    return pil_img


cutout = TransformT('Cutout', _cutout_pil_impl)


def _enhancer_impl(enhancer):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of PIL."""

    def impl(pil_img, level):
        v = float_parameter(level, 1.8) + .1  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


color = TransformT('Color', _enhancer_impl(ImageEnhance.Color))
contrast = TransformT('Contrast', _enhancer_impl(ImageEnhance.Contrast))
brightness = TransformT('Brightness', _enhancer_impl(ImageEnhance.Brightness))
sharpness = TransformT('Sharpness', _enhancer_impl(ImageEnhance.Sharpness))
hsv_h = TransformT('hsv_h', _hsv_h)
hsv_s = TransformT('hsv_s', _hsv_s)
hsv_v = TransformT('hsv_v', _hsv_v)
hed_h = TransformT('hed_h', _hed_h)
hed_e = TransformT('hed_e', _hed_e)
hed_d = TransformT('hed_d', _hed_d)
gauss_blur = TransformT('gauss_blur',_gauss_blur)
scaling = TransformT('scaling',_scaling)
gauss_noise = TransformT('gauss_noise',_gauss_noise)
elastic = TransformT('elastic',_elastic)

ALL_TRANSFORMS = [
    flip_lr, flip_ud, auto_contrast, equalize, invert, rotate, posterize,
    crop_bilinear, solarize, color, contrast, brightness, sharpness, shear_x,
    shear_y, translate_x, translate_y, cutout, blur, smooth,  hed_h, hed_e, hed_d, hsv_h, hsv_s, hsv_v, elastic, gauss_noise, scaling, gauss_blur
]

NAME_TO_TRANSFORM = {t.name: t for t in ALL_TRANSFORMS}
TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
