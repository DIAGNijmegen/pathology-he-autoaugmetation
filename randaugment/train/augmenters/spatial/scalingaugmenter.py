"""
This file contains a class for augmenting patches from whole slide images with scaling.
"""

from . import spatialaugmenterbase as dptspatialaugmenterbase

#from ...errors import augmentationerrors as dptaugmentationerrors

import numpy as np
import scipy.ndimage
import math

#----------------------------------------------------------------------------------------------------

class ScalingAugmenter(dptspatialaugmenterbase.SpatialAugmenterBase):
    """Apply scaling on the patch."""

    def __init__(self, scaling_range, interpolation_order=1):
        """
        Initialize the object.

        Args:
            scaling_range (tuple): Range for scaling factor selection. For example (0.8, 1.2).
            interpolation_order (int): Interpolation order from the range [0, 5].

        Raises:
            InvalidScalingRangeError: The sigma range for scaling is not valid.
            InvalidScalingInterpolationOrderError: The interpolation order for scaling is not valid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='scaling')

        # Initialize members.
        #
        self.__scaling_range = []       # Configured scaling range.
        self.__scaling_factor = None    # Current scaling factor to use.
        self.__interpolation_order = 0  # Interpolation order.

        # Save configuration.
        #
        self.__setscalingrange(scaling_range=scaling_range, interpolation_order=interpolation_order)

    def __setscalingrange(self, scaling_range, interpolation_order):
        """
        Set the scaling interval.

        Args:
            scaling_range (tuple): Range for scaling factor selection.
            interpolation_order (int): Interpolation order.

        Raises:
            InvalidScalingRangeError: The sigma range for scaling is not valid.
            InvalidScalingInterpolationOrderError: The interpolation order for scaling is not valid.
        """

        # Check the interval.
        #
        '''
        if len(scaling_range) != 2 or scaling_range[1] < scaling_range[0] or scaling_range[0] <= 0.0:
            raise dptaugmentationerrors.InvalidScalingRangeError(scaling_range)

        # Check the interpolation order.
        #
        if interpolation_order < 0 or 5 < interpolation_order:
            raise dptaugmentationerrors.InvalidScalingInterpolationOrderError(interpolation_order)
        '''
        # Store the setting.
        #
        self.__scaling_interval = list(scaling_range)
        self.__scaling_factor = scaling_range[0]
        self.__interpolation_order = int(interpolation_order)

    def shapes(self, target_shapes):
        """
        Calculate the required shape of the input to achieve the target output shape.

        Args:
            target_shapes (dict): Target output shape per level.

        Returns:
            (dict): Required input shape per level.
        """

        # Calculate the required input shape for each level.
        #
        return {level: (math.ceil(target_shapes[level][0] / self.__scaling_interval[0]), math.ceil(target_shapes[level][1] / self.__scaling_interval[0])) for level in target_shapes}

    def transform(self, patch):
        """
        Scale the patch with a random factor.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        # Pad patch to keep the original shape.
        #
        if self.__scaling_factor < 1.0:
            pad_ratio = ((1.0 / self.__scaling_factor - 1.0) / 2.0)
            pad_widths = (patch.shape[1] * pad_ratio, patch.shape[2] * pad_ratio)
            pad_config = ((0, 0), (math.ceil(pad_widths[0]), math.ceil(pad_widths[0])), (math.ceil(pad_widths[1]), math.ceil(pad_widths[1])))

            patch_padded = np.pad(array=patch, pad_width=pad_config, mode='reflect')
        else:
            patch_padded = patch

        # Zoom patch.
        #
        patch_transformed = scipy.ndimage.zoom(input=patch_padded, zoom=(1.0, self.__scaling_factor, self.__scaling_factor), order=self.__interpolation_order, mode='reflect')

        # Crop zoomed patch.
        #
        if patch_transformed.shape != patch.shape:
            border = (math.floor((patch_transformed.shape[1] - patch.shape[1]) / 2.0), math.floor((patch_transformed.shape[2] - patch.shape[2]) / 2.0))
            patch_transformed = patch_transformed[:, border[0]:border[0]+patch.shape[1], border[1]:border[1]+patch.shape[2]]

        return patch_transformed

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize the scaling factor.
        #
        self.__scaling_factor = np.random.uniform(low=self.__scaling_interval[0], high=self.__scaling_interval[1], size=None)
