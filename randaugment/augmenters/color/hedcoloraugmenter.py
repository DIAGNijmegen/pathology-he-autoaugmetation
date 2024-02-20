"""
This file contains a class for augmenting patches from whole slide images by applying color correction in HED color space.
"""
from . import coloraugmenterbase as dptcoloraugmenterbase


import numpy as np
import numpy as np
from scipy import linalg
from skimage.util import dtype, dtype_limits
from skimage.exposure import rescale_intensity
import time

rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0.27, 0.57, 0.78]]).astype('float32')
hed_from_rgb = linalg.inv(rgb_from_hed).astype('float32')


def rgb2hed(rgb):

    return separate_stains(rgb, hed_from_rgb)

def hed2rgb(hed):

    return combine_stains(hed, rgb_from_hed)

def separate_stains(rgb, conv_matrix):

    # # t = time.time()
    # rgb = dtype.img_as_float(rgb, force_copy=True).astype('float32')
    # # print('{f} took {s} s'.format(f='separate img_as_float', s=(time.time() - t)), flush=True)
    # # print('rgb type is {r}, matrix type is {m}'.format(r=rgb.dtype, m=conv_matrix.dtype), flush=True)
    #
    # # t = time.time()
    # rgb = -np.log(rgb)
    # # print('{f} took {s} s'.format(f='separate np.log', s=(time.time() - t)), flush=True)
    #
    # # t = time.time()
    # rgb += 2
    # rgb = np.reshape(rgb, (-1, 3))
    # # print('{f} took {s} s'.format(f='separate add reshape', s=(time.time() - t)), flush=True)
    #
    # # print('x shape is {s}, conv_matrix shape is {c}'.format(s=x.shape, c=conv_matrix.shape))
    #
    # # t = time.time()
    # stains = np.dot(rgb, conv_matrix)
    # # print('{f} took {s} s'.format(f='separate np.dot', s=(time.time() - t)), flush=True)
    #
    # return np.reshape(stains, rgb.shape)

    rgb = dtype.img_as_float(rgb, force_copy=True).astype('float32')
    rgb += 2
    stains = np.dot(np.reshape(-np.log(rgb), (-1, 3)), conv_matrix)
    return np.reshape(stains, rgb.shape)


def combine_stains(stains, conv_matrix):

    # # t = time.time()
    # stains = dtype.img_as_float(stains).astype('float32')
    # # stains = stains.astype('float32')
    # # conv_matrix = conv_matrix.astype('float32')
    # # print('{f} took {s} s'.format(f='separate img_as_float', s=(time.time() - t)), flush=True)
    # # print('stains type is {r}, matrix type is {m}'.format(r=stains.dtype, m=conv_matrix.dtype), flush=True)
    #
    # stains = -np.reshape(stains, (-1, 3))
    # # print('x shape is {s}, conv_matrix shape is {c}'.format(s=x.shape, c=conv_matrix.shape))
    #
    # # t = time.time()
    # logrgb2 = np.dot(stains, conv_matrix)
    # # print('{f} took {s} s'.format(f='combine np.dot', s=(time.time() - t)), flush=True)
    #
    # # t = time.time()
    # rgb2 = np.exp(logrgb2)
    # # print('{f} took {s} s'.format(f='combine np.exp', s=(time.time() - t)), flush=True)
    #
    # t = time.time()
    # x = rescale_intensity(np.reshape(rgb2 - 2, stains.shape),
    #                          in_range=(-1, 1))
    # print('{f} took {s} s'.format(f='combine rescale_intensity', s=(time.time() - t)), flush=True)
    #
    # return x

    stains = dtype.img_as_float(stains.astype('float64')).astype('float32')  # stains are out of range [-1, 1] so dtype.img_as_float complains if not float64
    logrgb2 = np.dot(-np.reshape(stains, (-1, 3)), conv_matrix)
    rgb2 = np.exp(logrgb2)
    return rescale_intensity(np.reshape(rgb2 - 2, stains.shape),
                             in_range=(-1, 1))

#----------------------------------------------------------------------------------------------------

class HedColorAugmenter(dptcoloraugmenterbase.ColorAugmenterBase):
    """Apply color correction in HED color space on the RGB patch."""

    def __init__(self, haematoxylin_sigma_range, haematoxylin_bias_range, eosin_sigma_range, eosin_bias_range, dab_sigma_range, dab_bias_range, cutoff_range):
        """
        Initialize the object. For each channel the augmented value is calculated as value = value * sigma + bias

        Args:
            haematoxylin_sigma_range (tuple, None): Adjustment range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.1, 0.1).
            haematoxylin_bias_range (tuple, None): Bias range for the Haematoxylin channel from the [-1.0, 1.0] range where 0.0 means no change. For example (-0.2, 0.2).
            eosin_sigma_range (tuple, None): Adjustment range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
            eosin_bias_range (tuple, None) Bias range for the Eosin channel from the [-1.0, 1.0] range where 0.0 means no change.
            dab_sigma_range (tuple, None): Adjustment range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
            dab_bias_range (tuple, None): Bias range for the DAB channel from the [-1.0, 1.0] range where 0.0 means no change.
            cutoff_range (tuple, None): Patches with mean value outside the cutoff interval will not be augmented. Values from the [0.0, 1.0] range. The RGB channel values are from the same range.

        Raises:
            InvalidHaematoxylinSigmaRangeError: The sigma range for Haematoxylin channel adjustment is not valid.
            InvalidHaematoxylinBiasRangeError: The bias range for Haematoxylin channel adjustment is not valid.
            InvalidEosinSigmaRangeError: The sigma range for Eosin channel adjustment is not valid.
            InvalidEosinBiasRangeError: The bias range for Eosin channel adjustment is not valid.
            InvalidDabSigmaRangeError: The sigma range for DAB channel adjustment is not valid.
            InvalidDabBiasRangeError: The bias range for DAB channel adjustment is not valid.
            InvalidCutoffRangeError: The cutoff range is not valid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='hed_color')

        # Initialize members.
        #
        self.__sigma_ranges = None  # Configured sigma ranges for H, E, and D channels.
        self.__bias_ranges = None   # Configured bias ranges for H, E, and D channels.
        self.__cutoff_range = None  # Cutoff interval.
        self.__sigmas = None        # Randomized sigmas for H, E, and D channels.
        self.__biases = None        # Randomized biases for H, E, and D channels.

        # Save configuration.
        #
        self.__setsigmaranges(haematoxylin_sigma_range=haematoxylin_sigma_range, eosin_sigma_range=eosin_sigma_range, dab_sigma_range=dab_sigma_range)
        self.__setbiasranges(haematoxylin_bias_range=haematoxylin_bias_range, eosin_bias_range=eosin_bias_range, dab_bias_range=dab_bias_range)
        self.__setcutoffrange(cutoff_range=cutoff_range)

    def __setsigmaranges(self, haematoxylin_sigma_range, eosin_sigma_range, dab_sigma_range):
        """
        Set the sigma intervals.

        Args:
            haematoxylin_sigma_range (tuple, None): Adjustment range for the Haematoxylin channel.
            eosin_sigma_range (tuple, None): Adjustment range for the Eosin channel.
            dab_sigma_range (tuple, None): Adjustment range for the DAB channel.

        Raises:
            InvalidHaematoxylinSigmaRangeError: The sigma range for Haematoxylin channel adjustment is not valid.
            InvalidEosinSigmaRangeError: The sigma range for Eosin channel adjustment is not valid.
            InvalidDabSigmaRangeError: The sigma range for DAB channel adjustment is not valid.
        """

        # Check the intervals.
        #
        '''
        if haematoxylin_sigma_range is not None:
            if len(haematoxylin_sigma_range) != 2 or haematoxylin_sigma_range[1] < haematoxylin_sigma_range[0] or haematoxylin_sigma_range[0] < -1.0 or 1.0 < haematoxylin_sigma_range[1]:
                raise dptaugmentationerrors.InvalidHaematoxylinSigmaRangeError(haematoxylin_sigma_range)

        if eosin_sigma_range is not None:
            if len(eosin_sigma_range) != 2 or eosin_sigma_range[1] < eosin_sigma_range[0] or eosin_sigma_range[0] < -1.0 or 1.0 < eosin_sigma_range[1]:
                raise dptaugmentationerrors.InvalidEosinSigmaRangeError(eosin_sigma_range)

        if dab_sigma_range is not None:
            if len(dab_sigma_range) != 2 or dab_sigma_range[1] < dab_sigma_range[0] or dab_sigma_range[0] < -1.0 or 1.0 < dab_sigma_range[1]:
                raise dptaugmentationerrors.InvalidDabSigmaRangeError(dab_sigma_range)
        '''
        # Store the settings.
        #
        self.__sigma_ranges = [haematoxylin_sigma_range, eosin_sigma_range, dab_sigma_range]

        self.__sigmas = [haematoxylin_sigma_range if haematoxylin_sigma_range is not None else 0.0,
                         eosin_sigma_range if eosin_sigma_range is not None else 0.0,
                         dab_sigma_range if dab_sigma_range is not None else 0.0]
        

    def __setbiasranges(self, haematoxylin_bias_range, eosin_bias_range, dab_bias_range):
        """
        Set the bias intervals.

        Args:
            haematoxylin_bias_range (tuple, None): Bias range for the Haematoxylin channel.
            eosin_bias_range (tuple, None) Bias range for the Eosin channel.
            dab_bias_range (tuple, None): Bias range for the DAB channel.

        Raises:
            InvalidHaematoxylinBiasRangeError: The bias range for Haematoxylin channel adjustment is not valid.
            InvalidEosinBiasRangeError: The bias range for Eosin channel adjustment is not valid.
            InvalidDabBiasRangeError: The bias range for DAB channel adjustment is not valid.
        """

        # Check the intervals.
        #
        '''
        if haematoxylin_bias_range is not None:
            if len(haematoxylin_bias_range) != 2 or haematoxylin_bias_range[1] < haematoxylin_bias_range[0] or haematoxylin_bias_range[0] < -1.0 or 1.0 < haematoxylin_bias_range[1]:
                raise dptaugmentationerrors.InvalidHaematoxylinBiasRangeError(haematoxylin_bias_range)

        if eosin_bias_range is not None:
            if len(eosin_bias_range) != 2 or eosin_bias_range[1] < eosin_bias_range[0] or eosin_bias_range[0] < -1.0 or 1.0 < eosin_bias_range[1]:
                raise dptaugmentationerrors.InvalidEosinBiasRangeError(eosin_bias_range)

        if dab_bias_range is not None:
            if len(dab_bias_range) != 2 or dab_bias_range[1] < dab_bias_range[0] or dab_bias_range[0] < -1.0 or 1.0 < dab_bias_range[1]:
                raise dptaugmentationerrors.InvalidDabBiasRangeError(dab_bias_range)
        '''
        # Store the settings.
        #
        self.__bias_ranges = [haematoxylin_bias_range, eosin_bias_range, dab_bias_range]

        self.__biases = [haematoxylin_bias_range if haematoxylin_bias_range is not None else 0.0,
                         eosin_bias_range if eosin_bias_range is not None else 0.0,
                         dab_bias_range if dab_bias_range is not None else 0.0]
        
    def __setcutoffrange(self, cutoff_range):
        """
        Set the cutoff value. Patches with mean value outside the cutoff interval will not be augmented.

        Args:
            cutoff_range (tuple, None): Patches with mean value outside the cutoff interval will not be augmented.

        Raises:
            InvalidCutoffRangeError: The cutoff range is not valid.
        """

        # Check the interval.
        #
       
        # Store the setting.
        #
        self.__cutoff_range = cutoff_range if cutoff_range is not None else [0.0, 1.0]

    def transform(self, patch):
        """
        Apply color deformation on the patch.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """
        #print('hed self.__biases',self.__biases)
        #print('hed self.__sigmas',self.__sigmas)

        # Check if the patch is inside the cutoff values.
        #
        patch_mean = np.mean(a=patch) / 255.0
        if self.__cutoff_range[0] <= patch_mean <= self.__cutoff_range[1]:
            # Reorder the patch to channel last format and convert the image patch to HED color coding.
            #
            #patch_image = np.transpose(a=patch, axes=(1, 2, 0))
            patch_hed = rgb2hed(rgb=patch)

            # Augment the Haematoxylin channel.
            #
            if self.__sigmas[0] != 0.0:
                patch_hed[:, :, 0] *= (1.0 + self.__sigmas[0])

            if self.__biases[0] != 0.0:
                patch_hed[:, :, 0] += self.__biases[0]

            # Augment the Eosin channel.
            #
            if self.__sigmas[1] != 0.0:
                patch_hed[:, :, 1] *= (1.0 + self.__sigmas[1])

            if self.__biases[1] != 0.0:
                patch_hed[:, :, 1] += self.__biases[1]

            # Augment the DAB channel.
            #
            if self.__sigmas[2] != 0.0:
                patch_hed[:, :, 2] *= (1.0 + self.__sigmas[2])

            if self.__biases[2] != 0.0:
                patch_hed[:, :, 2] += self.__biases[2]

            # Convert back to RGB color coding and order back to channels first order.
            #
            patch_rgb = hed2rgb(hed=patch_hed)
            patch_rgb = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)
            patch_rgb *= 255.0
            patch_rgb = patch_rgb.astype(dtype=np.uint8)

            #patch_transformed = np.transpose(a=patch_rgb, axes=(2, 0, 1))

            return patch_rgb

        else:
            # The image patch is outside the cutoff interval.
            #
            return patch


    # def transform(self, patch):
    #     """
    #     Apply color deformation on the patch.
    #
    #     Args:
    #         patch (np.ndarray): Patch to transform.
    #
    #     Returns:
    #         np.ndarray: Transformed patch.
    #     """
    #     import time
    #
    #     print('### Timing ###')
    #     t_init = time.time()
    #
    #     # Check if the patch is inside the cutoff values.
    #     #
    #     patch_mean = np.mean(a=patch) / 255.0
    #     if self.__cutoff_range[0] <= patch_mean <= self.__cutoff_range[1]:
    #         # Reorder the patch to channel last format and convert the image patch to HED color coding.
    #         #
    #         # t = time.time()
    #         patch_image = np.transpose(a=patch, axes=(1, 2, 0))
    #         # print('{f} took {s} s'.format(f='initial transpose', s=(time.time() - t)), flush=True)
    #
    #         t = time.time()
    #         patch_hed = rgb2hed(rgb=patch_image)
    #         print('{f} took {s} s'.format(f='rgb2hed', s=(time.time() - t)), flush=True)
    #
    #         # Augment the Haematoxylin channel.
    #         #
    #         # t = time.time()
    #         if self.__sigmas[0] != 0.0:
    #             patch_hed[:, :, 0] *= (1.0 + self.__sigmas[0])
    #
    #         if self.__biases[0] != 0.0:
    #             patch_hed[:, :, 0] += self.__biases[0]
    #         # print('{f} took {s} s'.format(f='H variation', s=(time.time() - t)), flush=True)
    #
    #         # Augment the Eosin channel.
    #         #
    #         # t = time.time()
    #         if self.__sigmas[1] != 0.0:
    #             patch_hed[:, :, 1] *= (1.0 + self.__sigmas[1])
    #
    #         if self.__biases[1] != 0.0:
    #             patch_hed[:, :, 1] += self.__biases[1]
    #         # print('{f} took {s} s'.format(f='E variation', s=(time.time() - t)), flush=True)
    #
    #         # Augment the DAB channel.
    #         #
    #         # t = time.time()
    #         if self.__sigmas[2] != 0.0:
    #             patch_hed[:, :, 2] *= (1.0 + self.__sigmas[2])
    #
    #         if self.__biases[2] != 0.0:
    #             patch_hed[:, :, 2] += self.__biases[2]
    #         # print('{f} took {s} s'.format(f='D variation', s=(time.time() - t)), flush=True)
    #
    #         # Convert back to RGB color coding and order back to channels first order.
    #         #
    #         t = time.time()
    #         patch_rgb = hed2rgb(hed=patch_hed)
    #         print('{f} took {s} s'.format(f='hed2rgb', s=(time.time() - t)), flush=True)
    #
    #         # t = time.time()
    #         patch_rgb = np.clip(a=patch_rgb, a_min=0.0, a_max=1.0)
    #         # print('{f} took {s} s'.format(f='clip', s=(time.time() - t)), flush=True)
    #
    #         # t = time.time()
    #         patch_rgb *= 255.0
    #         patch_rgb = patch_rgb.astype(dtype=np.uint8)
    #         # print('{f} took {s} s'.format(f='255 and uint8', s=(time.time() - t)), flush=True)
    #
    #         # t = time.time()
    #         patch_transformed = np.transpose(a=patch_rgb, axes=(2, 0, 1))
    #         # print('{f} took {s} s'.format(f='transpose end', s=(time.time() - t)), flush=True)
    #
    #         p = patch_transformed
    #
    #     else:
    #         # The image patch is outside the cutoff interval.
    #         #
    #         p = patch
    #
    #     print('{f} took {s} s'.format(f='all', s=(time.time() - t_init)), flush=True)
    #     return p

   
