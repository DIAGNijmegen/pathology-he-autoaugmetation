"""
Keras data generators for supervised classification tasks.
"""
from model.data_augmentation import rgb_to_gray
from utils import dl
from os.path import join, exists, basename, dirname, splitext
import numpy as np
import argparse
import os
import gc
import shutil
from glob import glob
import time

#----------------------------------------------------------------------------------------------------

class SupervisedGenerator(object):
    """
    Class to randomly provide batches of supervised images loaded from numpy arrays on disk.
    """

    def __init__(self, x_path, y_path, batch_size, augmenter, one_hot=True, compare_augmentation=False, color_space='rgb'):
        """
        Class to randomly provide batches of images loaded from numpy arrays on disk.

        Args:
            x_path (str): path to array of images in numpy uint8 format with channels in last dimension.
            y_path (str): path to array of labels.
            batch_size (int): number of samples per batch.
            augment (bool): True to apply rotation and flipping augmentation.
        """

        # Params
        self.batch_size = batch_size
        self.x_path = x_path
        self.y_path = y_path
        self.augmenter = augmenter
        self.one_hot = one_hot
        self.compare_augmentation = compare_augmentation
        self.color_space = color_space

        # Read data
        self.x = np.load(x_path)  # need to read here due to keras multiprocessing
        # self.x = None
        self.y = np.load(y_path)

        # Drop 255 class
        self.idx = np.where(self.y != 255)[0]

        self.classes = np.unique(self.y[self.idx])
        self.n_classes = len(self.classes)
        self.n_samples = len(self.idx)
        self.n_batches = self.n_samples // self.batch_size

        # Indexes for classes
        self.class_idx = []
        for i in self.classes:
            self.class_idx.append(
                np.where(self.y == i)[0]
            )

    def get_n_classes(self):
        return self.n_classes

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        return self.n_batches

    def augment_batch(self, x):
        """
        Applies augmentation.

        Args:
            x: batch of images with shape [batch, x, y, c].

        Returns: batch of augmented images.

        """

        if self.augmenter is not None:
            x_augmented = np.zeros_like(x)
            for i in range(x.shape[0]):
                x_augmented[i, ...] = self.augmenter.augment(x[i, ...])
        else:
            x_augmented = x

        if self.compare_augmentation:
            x_augmented = np.dstack([x, x_augmented])

        return x_augmented

    def get_batch(self):
        """
        Draws a random set of samples from the training set and assembles pairs of images and labels.

        Returns: batch of images with shape [batch, x, y, c].
        """

        # Get samples
        idxs = []
        for i in range(self.n_classes):
            idxs.append(
                np.random.choice(self.class_idx[i], self.batch_size // self.n_classes, replace=True)
            )

        # Merge
        idxs = np.concatenate(idxs)

        # Randomize
        np.random.shuffle(idxs)

        # Build batch
        if self.x is None:
            self.x = np.load(self.x_path)
        x = self.x[idxs, ...]
        y = self.y[idxs]

        # Color space
        if self.color_space == 'grayscale':
            x = rgb_to_gray(x)

        # Augment
        if self.augmenter is not None:
            # t = time.time()
            x = self.augment_batch(x)
            # print('{f} took {s} s'.format(f='augment_batch', s=(time.time() - t)), flush=True)

        # Range (float [-1, +1])
        x = (x / 255.0 * 2) - 1

        # One-hot encoding
        if self.one_hot:
            y = np.eye(self.n_classes)[y]

        return x, y

    def next(self):
        # t = time.time()
        batch = self.get_batch()
        # print('{f} took {s} s'.format(f='next', s=(time.time() - t)), flush=True)
        batch = self.transform(batch)
        return batch

    def transform(self, batch):
        """
        Implement this function to alter the returned batch in any way. Useful if you inherit this class to transform
        the batch data.

        Args:
            batch: batch of images with shape [batch, x, y, c].

        Returns: batch of images with shape [batch, x, y, c].
        """
        return batch

#----------------------------------------------------------------------------------------------------

class SupervisedSequence(dl.utils.Sequence):
    """
    Class to sequentially provide batches of supervised images loaded from numpy arrays on disk.
    """

    def __init__(self, x_path, y_path, batch_size, one_hot=True, color_space='rgb', include_255=False, augmenter=None, compare_augmentation=False):
        """
        Class to sequentially provide batches of supervised images loaded from numpy arrays on disk.

        Args:
            x_path (str): path to array of images in numpy uint8 format with channels in last dimension.
            y_path (str): path to array of labels.
            batch_size (int): number of samples per batch.
        """

        # Params
        self.batch_size = batch_size
        self.x_path = x_path
        self.y_path = y_path
        self.one_hot = one_hot
        self.color_space = color_space
        self.include_255 = include_255
        self.augmenter = augmenter
        self.compare_augmentation = compare_augmentation

        # Read data
        self.x = np.load(x_path)  # need to read here due to keras multiprocessing
        self.y = np.load(y_path)

        # Drop 255 class
        if self.include_255:
            self.idx = np.arange(0, self.y.shape[0])
        else:
            self.idx = np.where(self.y != 255)[0]

        self.n_samples = len(self.idx)
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))
        self.classes = np.unique(self.y[self.idx])
        self.n_classes = len(self.classes)

    def augment_batch(self, x):
        """
        Applies augmentation.

        Args:
            x: batch of images with shape [batch, x, y, c].

        Returns: batch of augmented images.

        """

        if self.augmenter is not None:
            x_augmented = np.zeros_like(x)
            for i in range(x.shape[0]):
                x_augmented[i, ...] = self.augmenter.augment(x[i, ...])
        else:
            x_augmented = x

        if self.compare_augmentation:
            x_augmented = np.dstack([x, x_augmented])

        return x_augmented

    def get_n_classes(self):
        return self.n_classes

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        return self.n_batches

    def get_batch(self, idx):
        """
        Draws a set of samples from the dataset based on the index and assembles pairs of images and labels. Index refers
        to batches (not samples).

        Returns: batch of images with shape [batch, x, y, c].
        """

        # Get samples
        idx_batch = idx * self.batch_size
        if idx_batch + self.batch_size >= self.n_samples:
            idxs = np.arange(idx_batch, self.n_samples)
        else:
            idxs = np.arange(idx_batch, idx_batch + self.batch_size)

        # Build batch
        if self.x is None:
            self.x = np.load(self.x_path)
        x = self.x[self.idx[idxs], ...]
        y = self.y[self.idx[idxs]]

        # Color space
        if self.color_space == 'grayscale':
            x = rgb_to_gray(x)

        # Augment
        if self.augmenter is not None:
            x = self.augment_batch(x)

        # Format
        x = (x / 255.0 * 2) - 1

        # One-hot encoding
        if self.one_hot:
            y = np.eye(self.n_classes)[y]

        return x, y

    def __getitem__(self, idx):
        batch = self.get_batch(idx)
        batch = self.transform(batch)
        return batch

    def transform(self, batch):
        """
        Implement this function to alter the returned batch in any way. Useful if you inherit this class to transform
        the batch data.

        Args:
            batch: batch of images with shape [batch, x, y, c].

        Returns: batch of images with shape [batch, x, y, c].
        """
        return batch

    def get_all_labels(self, one_hot=True):

        y = self.y[self.idx]
        if one_hot:
            y = np.eye(self.n_classes)[y]

        return y


#----------------------------------------------------------------------------------------------------

class NumpyArrayManager(object):

    def __init__(self, x_path, y_path, ignore_255, reload_ratio=0.5):

        # Params
        self.x_path = x_path
        self.y_path = y_path
        self.ignore_255 = ignore_255
        self.reload_ratio = reload_ratio
        self.x = None
        self.y = None
        self.current_part = None
        self.random_samples_read = None
        self.class_idx = None
        # self.current_idx = None

        # Get info
        x_tag = splitext(basename(x_path))[0]
        y_tag = splitext(basename(y_path))[0]
        self.x_pattern = join(dirname(x_path), x_tag + '_parts', x_tag + '_*.npy')
        self.y_pattern = join(dirname(y_path), x_tag + '_parts', y_tag + '_*.npy')
        self.x_paths = sorted(glob(self.x_pattern))
        self.y_paths = sorted(glob(self.y_pattern))
        self.n_parts = len(self.x_paths)
        if self.n_parts <= 0:
            raise NotImplementedError('Numpy array parts not found in {p} and {s}.'.format(p=self.x_pattern, s=self.y_pattern))

        # Global info about classes
        y = np.load(y_path)
        if ignore_255:
            self.idx = np.where(y != 255)[0]
        else:
            self.idx = np.arange(0, y.shape[0])
        self.classes = np.unique(y[self.idx])
        self.n_classes = len(self.classes)
        self.len = len(y)

        # Read first part
        self.read_part(i=0)

    def read_part(self, i):

        print('Reading part {i}...'.format(i=i), flush=True)

        self.current_part = i
        self.x = np.load(self.x_paths[i])
        self.y = np.load(self.y_paths[i])
        self.random_samples_read = 0

        if i == 0:
            self.available_idx = [0, self.x.shape[0]]
            # self.current_idx = 0
        else:
            # self.current_idx += self.available_idx[1]
            self.available_idx = [self.available_idx[1], self.available_idx[1] + self.x.shape[0]]

        # Indexes for classes
        self.class_idx = []
        for i in self.classes:
            self.class_idx.append(
                np.where(self.y == i)[0]
            )

    def sample_random(self, n_samples):

        # Get samples
        idxs = []
        for i in range(self.n_classes):
            idxs.append(
                np.random.choice(self.class_idx[i], n_samples // self.n_classes, replace=True)
            )

        # Merge
        idxs = np.concatenate(idxs)

        # Randomize
        np.random.shuffle(idxs)

        # Build batch
        x = self.x[idxs, ...]
        y = self.y[idxs, ...]

        # Load next part if needed
        self.random_samples_read += n_samples
        if self.random_samples_read >= int(self.reload_ratio * self.x.shape[0]):
            self.read_part(i=self.current_part + 1)

        return x, y

    def __getitem__(self, index):

        # Params
        idx_start = index[0].start
        idx_stop = index[0].stop

        # Cases
        if idx_start >= self.available_idx[0]:

            if idx_start < self.available_idx[1]:
                pass
            else:
                self.read_part(i=self.current_part + 1)

            if idx_stop <= self.available_idx[1]:
                x = self.x[idx_start-self.available_idx[0]:idx_stop-self.available_idx[0], ...]
                y = self.y[idx_start-self.available_idx[0]:idx_stop-self.available_idx[0], ...]

            else:
                aux = self.available_idx[1]
                x = self.x[idx_start-self.available_idx[0]:aux-self.available_idx[0], ...]
                y = self.y[idx_start-self.available_idx[0]:aux-self.available_idx[0], ...]

                self.read_part(i=self.current_part + 1)

                x = np.concatenate([x, self.x[aux-self.available_idx[0]: idx_stop-self.available_idx[0], ...]], axis=-1)
                y = np.concatenate([y, self.y[aux-self.available_idx[0]: idx_stop-self.available_idx[0], ...]], axis=-1)
        else:
            raise Exception('idx start {i} cannot be smaller than available index {j}'.format(i=idx_start, j=self.available_idx[0]))

        return x, y

    def __len__(self):
        return self.len

#----------------------------------------------------------------------------------------------------

class AugmenterGenerator(object):
    """
    Class to randomly provide batches of supervised images loaded from numpy arrays on disk.
    """

    def __init__(self, x_path, y_path, batch_size, augmenter, augmenter_stain, compare_augmentation=False, color_space='rgb', prob_white_patch=None):
        """
        Class to randomly provide batches of images loaded from numpy arrays on disk.

        Args:
            x_path (str): path to array of images in numpy uint8 format with channels in last dimension.
            y_path (str): path to array of labels.
            batch_size (int): number of samples per batch.
            augment (bool): True to apply rotation and flipping augmentation.
        """

        # Params
        self.batch_size = batch_size
        self.x_path = x_path
        self.y_path = y_path
        self.augmenter = augmenter
        self.augmenter_stain = augmenter_stain
        self.compare_augmentation = compare_augmentation
        self.color_space = color_space
        self.prob_white_patch = prob_white_patch

        # Read data
        self.x = np.load(x_path)  # need to read here due to keras multiprocessing
        self.y = np.load(y_path)

        # Drop 255 class
        self.idx = np.where(self.y != 255)[0]

        # White patches
        if self.prob_white_patch is not None:
            self.prob_white_patch = float(self.prob_white_patch)
            n_white = np.int(len(self.y) * self.prob_white_patch)
            n_normal = len(self.y) - n_white
            self.idx_white = np.concatenate([np.ones(n_white), np.zeros(n_normal)])
            np.random.shuffle(self.idx_white)
        else:
            self.idx_white = None

        self.classes = np.unique(self.y[self.idx])
        # self.n_classes = len(self.classes)
        self.n_samples = len(self.idx)
        self.n_batches = self.n_samples // self.batch_size

        # # Indexes for classes
        # self.class_idx = []
        # for i in self.classes:
        #     self.class_idx.append(
        #         np.where(self.y == i)[0]
        #     )

    # def get_n_classes(self):
    #     return self.n_classes

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        return self.n_batches

    def augment_batch(self, x, augmenter, compare_augmentation):
        """
        Applies augmentation.

        Args:
            x: batch of images with shape [batch, x, y, c].

        Returns: batch of augmented images.

        """

        if augmenter is not None:
            x_augmented = np.zeros_like(x)
            for i in range(x.shape[0]):
                x_augmented[i, ...] = augmenter.augment(x[i, ...])
        else:
            x_augmented = x

        if compare_augmentation:
            x_augmented = np.dstack([x, x_augmented])

        return x_augmented

    def get_batch(self):
        """
        Draws a random set of samples from the training set and assembles pairs of images and labels.

        Returns: batch of images with shape [batch, x, y, c].
        """

        # Get samples
        idxs = np.random.choice(self.idx, self.batch_size, replace=True)

        # Randomize
        np.random.shuffle(idxs)

        # Build batch
        if self.x is None:
            self.x = np.load(self.x_path)
        x = self.x[idxs, ...]

        # Add white patches
        # Exactly white 255 to make sure the network always reconstruct this tone
        if self.idx_white is not None:

            # White patch
            # x[self.idx_white[idxs] == 1, ...] = 255  # np.random.randint(250, 256, 1).astype('int32')

            # Per patch
            for idx in np.where(self.idx_white[idxs] == 1)[0]:

                # Sample
                x_patch = x[idx, ...]

                # Rotate
                x_patch = np.rot90(m=x_patch, k=np.random.randint(0, 4), axes=(0, 1))

                # Impose white rectangle
                w1 = np.random.randint(0, x_patch.shape[0]//3)
                h1 = np.random.randint(0, x_patch.shape[1]//3)
                w2 = np.random.randint(w1, x_patch.shape[0])
                h2 = np.random.randint(h1, x_patch.shape[1])
                x_patch[w1:w2, h1:h2, :] = 255

                # Rotate
                x_patch = np.rot90(m=x_patch, k=np.random.randint(0, 4), axes=(0, 1))

                # Store
                x[idx, ...] = x_patch

        # Baseline augmentation
        x = self.augment_batch(x, self.augmenter, compare_augmentation=False)

        # Color space
        if self.color_space == 'grayscale':
            x_color = rgb_to_gray(x)
        else:
            x_color = x

        # Advanced augmentation
        # t = time.time()
        aug_x = self.augment_batch(x_color, self.augmenter_stain, compare_augmentation=self.compare_augmentation)
        # print('{f} took {s} s'.format(f='augment_batch()', s=(time.time() - t)), flush=True)

        # Range (float [-1, +1])
        x = (x / 255.0 * 2) - 1
        aug_x = (aug_x / 255.0 * 2) - 1

        return aug_x, x

    def next(self):
        # t = time.time()
        batch = self.get_batch()
        # print('{f} took {s} s'.format(f='next()', s=(time.time() - t)), flush=True)

        batch = self.transform(batch)
        return batch

    def transform(self, batch):
        """
        Implement this function to alter the returned batch in any way. Useful if you inherit this class to transform
        the batch data.

        Args:
            batch: batch of images with shape [batch, x, y, c].

        Returns: batch of images with shape [batch, x, y, c].
        """
        return batch

#----------------------------------------------------------------------------------------------------

def change_range_less_memory(x, temp_dir, n_samples_per_chunk):

    # Dir
    if not exists(temp_dir):
        os.makedirs(temp_dir)

    # Iterate
    paths = []
    counter = 0
    i = 0
    for i in range(0, x.shape[0], n_samples_per_chunk):

        x_sample = x[i:i+n_samples_per_chunk, ...]
        x_sample = ((x_sample * 0.5 + 0.5) * 255).astype('uint8')
        output_path = join(temp_dir, 'chunk_{i}.npy'.format(i=i))
        np.save(output_path, x_sample)
        paths.append(output_path)
        counter += n_samples_per_chunk

        del x_sample
        gc.collect()

    # Remaining
    if counter < x.shape[0]:

        x_sample = x[counter:, ...]
        x_sample = ((x_sample * 0.5 + 0.5) * 255).astype('uint8')
        output_path = join(temp_dir, 'chunk_{i}.npy'.format(i=i+1))
        np.save(output_path, x_sample)
        paths.append(output_path)

    # Delete float
    del x
    gc.collect()

    # Read uint8
    x = np.concatenate([np.load(path) for path in paths], axis=0)

    # Delete temp
    if exists(temp_dir):
        shutil.rmtree(temp_dir)

    return x

#----------------------------------------------------------------------------------------------------

# if __name__ == '__main__':

    # nam = NumpyArrayManager(
    #     x_path=r"C:\Users\david\Downloads\data_augmentation_debug\validation_x.npy",
    #     y_path=r"C:\Users\david\Downloads\data_augmentation_debug\validation_y.npy",
    #     ignore_255=True,
    #     reload_ratio=0.5
    # )
    #
    # for i in range(0, 100000, 1000):
    #     print(i)
    #     x, y = nam[i:i+16, ...]
#
#     change_range_less_memory(
#         x=np.random.uniform(-1, 1, (5127, 128, 128, 3)),
#         temp_dir=r'C:\Users\david\Downloads\data_augmentation_debug\change_range\temp',
#         n_samples_per_chunk=500
#     )

#
#     def compute_on_batch(model, n_batches, dataset, output_dir):
#
#         x_data = []
#         y_data = []
#         pred_data = []
#         counter = 0
#         for x, y in dataset:
#             pred = model.predict_on_batch(x)
#             x_data.append(x)
#             y_data.append(y)
#             pred_data.append(pred)
#
#             counter += 1
#             if counter >= n_batches:
#                 break
#
#         x_data = np.concatenate(x_data, axis=0)
#         y_data = np.concatenate(y_data, axis=0)
#         pred_data = np.concatenate(pred_data, axis=0)
#
#         x_data = ((x_data * 0.5 + 0.5) * 255).astype('uint8')
#         y_data = ((y_data * 0.5 + 0.5) * 255).astype('uint8')
#         pred_data = ((pred_data * 0.5 + 0.5) * 255).astype('uint8')
#
#         np.save(join(output_dir, 'x_data.npy'), x_data)
#         np.save(join(output_dir, 'y_data.npy'), y_data)
#         np.save(join(output_dir, 'pred_data.npy'), pred_data)
#
#         from preparation.prepare_mitosis_dataset import dump_patches
#         dump_patches(
#             x_paths=[
#                 join(output_dir, 'x_data.npy'),
#                 join(output_dir, 'y_data.npy'),
#                 join(output_dir, 'pred_data.npy')
#             ],
#             y_path=None,
#             output_dir=output_dir,
#             max_items=1000,
#             encode=False
#         )
#
#
    # from model.data_augmentation import DataAugmenter
    # from matplotlib import pyplot as plt
    #
    # dataset = AugmenterGenerator(
    #     x_path=r"C:\Users\david\Downloads\data_augmentation_debug\validation_x.npy",
    #     y_path=r"C:\Users\david\Downloads\data_augmentation_debug\validation_y.npy",
    #     batch_size=64,
    #     augmenter=DataAugmenter(augmentation_tag='baseline'),
    #     # augmenter_stain=DataAugmenter(augmentation_tag='hsv_strong'),
    #     augmenter_stain=DataAugmenter(augmentation_tag='hsv_strong_extended3'),
    #     compare_augmentation=True,
    #     color_space='rgb',
    #     prob_white_patch=0.3
    # )
    #
    # for i in range(10):
    #     x, y = next(dataset)
    #     for j in range(x.shape[0]):
    #
    #         patch_aug = x[j, ...]
    #         patch_aug = patch_aug * 0.5 + 0.5
    #         plt.imsave(r'C:\Users\david\Downloads\data_augmentation_debug\white_rect_prob\batch_{b}_image_{im}.png'.format(b=i, im=j), patch_aug, vmin=0.0, vmax=1.0)

#
#     from evaluation import metrics
#     model = dl.models.load_model(r'Y:\projects\DataAugmentationComPat\sandbox\standardizer\unet.h5')
#
#     compute_on_batch(model, n_batches=16, dataset=dataset, output_dir=r'Y:\projects\DataAugmentationComPat\sandbox\standardizer\patches')
#


#
#     import gc
#
#     dataset_dir = r'C:\Users\david\Downloads\data_augmentation_debug'
#     batch_size = 32
#     validation_gen = SupervisedSequence(
#         x_path=join(dataset_dir, 'validation_x.npy'),
#         y_path=join(dataset_dir, 'validation_y.npy'),
#         batch_size=batch_size,
#         one_hot=True
#     )
#     del validation_gen
#     gc.collect()
#
#     validation_gen = SupervisedSequence(
#         x_path=join(dataset_dir, 'validation_x.npy'),
#         y_path=join(dataset_dir, 'validation_y.npy'),
#         batch_size=batch_size,
#         one_hot=True
#     )
#
#     print('hola')