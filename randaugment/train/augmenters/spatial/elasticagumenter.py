"""
This file contains a class for augmenting patches from whole slide images by applying elastic transformation.
"""

from . import spatialaugmenterbase as dptspatialaugmenterbase


import scipy.ndimage.interpolation
import scipy.ndimage.filters
import numpy as np

#----------------------------------------------------------------------------------------------------

class ElasticAugmenter(dptspatialaugmenterbase.SpatialAugmenterBase):
    """Apply elastic deformation to patch. Deformation maps are created when the first patch is deformed."""

    def __init__(self, sigma_interval, alpha_interval, map_count, interpolation_order=1):
        """
        Initialize the object.

        Args:
            sigma_interval (tuple): Interval for sigma selection for Gaussian filter map.
            alpha_interval (tuple): Interval for alpha selection for the severity of the deformation.
            map_count (int): Amount of deformation maps to precalculate.
            interpolation_order (int): Interpolation order from the range [0, 5].

        Raises:
            InvalidElasticSigmaIntervalError: The interval of sigma for elastic deformation is invalid.
            InvalidElasticAlphaIntervalError: The interval of alpha for elastic deformation is invalid.
            InvalidElasticMapCountError: The number of elastic deformation maps to precalculate is invalid.
            InvalidElasticInterpolationOrderError: The interpolation order for elastic transformation is not valid.
        """

        # Initialize base class.
        #
        super().__init__(keyword='elastic')

        # Initialize members.
        #
        self.__sigma_interval = []      # Sigma.
        self.__alpha_interval = []      # Alpha.
        self.__map_count = 0            # Number of deformation maps to pre-calculate.
        self.__interpolation_order = 0  # Interpolation order.
        self.__deformation_maps = {}    # Deformation maps per patch shape.
        self.__map_choice = 0           # Selected deformation map.

        # Save configuration.
        #
        self.__cofiguredeformationmaps(sigma_interval=sigma_interval, alpha_interval=alpha_interval, map_count=map_count, interpolation_order=interpolation_order)

    def __cofiguredeformationmaps(self, sigma_interval, alpha_interval, map_count, interpolation_order):
        """
        Configure the deformation map calculation parameters.

        Args:
            sigma_interval (tuple): Interval for sigma selection for Gaussian filter map.
            alpha_interval (tuple): Interval for alpha selection for the severity of the deformation.
            map_count (int): Amount of deformation maps to precalculate.
            interpolation_order (int): Interpolation order from the range [0, 5].

        Raises:
            InvalidElasticSigmaIntervalError: The interval of sigma for elastic deformation is invalid.
            InvalidElasticAlphaIntervalError: The interval of alpha for elastic deformation is invalid.
            InvalidElasticMapCountError: The number of elastic deformation maps to precalculate is invalid.
            InvalidElasticInterpolationOrderError: The interpolation order for elastic transformation is not valid.
        """

        # Check the sigma interval.
      

        # Store the settings.
        #
        self.__sigma_interval = list(sigma_interval)
        self.__alpha_interval = list(alpha_interval)
        self.__map_count = int(map_count)
        self.__interpolation_order = int(interpolation_order)

    def __createdeformationmaps(self, image_shape):
        """
        Elastic deformation of images as described in Simard, Steinkraus and Platt, "Best Practices for Convolutional Neural Networks applied to Visual Document Analysis",
        in Proc. of the International Conference on Document Analysis and Recognition, 2003.

        Args:
            image_shape (tuple): Image shape to deform.
        """

        self.__deformation_maps[image_shape[1:3]] = []

        for _ in range(self.__map_count):
            alpha = np.random.uniform(low=self.__alpha_interval[0], high=self.__alpha_interval[1], size=None)
            sigma = np.random.uniform(low=self.__sigma_interval[0], high=self.__sigma_interval[1], size=None)

            dx = scipy.ndimage.filters.gaussian_filter(input=(np.random.rand(*image_shape) * 2 - 1), sigma=sigma, mode='constant', cval=0) * alpha
            dy = scipy.ndimage.filters.gaussian_filter(input=(np.random.rand(*image_shape) * 2 - 1), sigma=sigma, mode='constant', cval=0) * alpha
            z, x, y = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), np.arange(image_shape[2]), indexing='ij')
            indices = (np.reshape(z, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)))

            self.__deformation_maps[image_shape[1:3]].append(indices)

    def transform(self, patch):
        """
        Deform the image with a random deformation map.

        Args:
            patch (np.ndarray): Patch to transform.

        Returns:
            np.ndarray: Transformed patch.
        """

        # Initialize the deformation maps.
        #
        if patch.shape[1:3] not in self.__deformation_maps:
            self.__createdeformationmaps(patch.shape)

        # Apply elastic deformation.
        #
        indices = self.__deformation_maps[patch.shape[1:3]][self.__map_choice]
        patch_transformed = scipy.ndimage.interpolation.map_coordinates(input=patch, coordinates=indices, order=self.__interpolation_order, mode='reflect').reshape(patch.shape)

        return patch_transformed

    def randomize(self):
        """Randomize the parameters of the augmenter."""

        # Randomize the transformation map.
        #
        self.__map_choice = np.random.randint(low=0, high=self.__map_count - 1)
