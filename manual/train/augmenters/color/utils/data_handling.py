import matplotlib as mpl
mpl.use('Agg')


import os
import shutil
from glob import glob
from os.path import join, exists, splitext, basename, dirname

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.misc
import gc

from utils.copy_files import cache_file


def dump_patches(x_paths, output_dir, y_path=None, max_items=None, encode=False, pred_path=None, exclude_255_y_path=None):
    """
    Reads patches from numpy arrays and save them to disk in PNG format.

    Args:
        x_paths (list): list of paths to numpy arrays (patches will be concatenated across these paths).
        output_dir (str): output directory to save PNGs.
        y_path (str): path to numpy array with labels.
        max_items (int or None): number of images to save.
        encode (bool): True to randomize image file names and store a key spreadsheet.

    """

    # Read images
    xs = [np.load(x_path) for x_path in x_paths]

    # Drop 255
    # 255 labels are excluded from the predictions and y_true (after label_map processing). Need to exclude them from x
    # using original labels.
    if exclude_255_y_path is not None:
        y_true_255 = np.load(exclude_255_y_path)
        xs = [x[y_true_255 != 255] for x in xs]

    # Number of images
    n_images = xs[0].shape[0]

    # Read labels
    if y_path:
        y = np.load(y_path)
    else:
        y = np.zeros(n_images)

    # Read predictions
    if pred_path:
        preds = np.load(pred_path)
    else:
        preds = np.zeros(n_images)

    # Make directory
    # output_patch_dir = join(output_dir, splitext(basename(y_path))[0])
    if not exists(output_dir):
        os.mkdir(output_dir)

    # Sample
    n_samples = int(np.min([max_items, n_images])) if max_items else n_images
    idx = np.random.choice(n_images, n_samples, replace=False)

    # Iterate through images
    print('Saving patches ...', flush=True)
    key = {}
    for i, i_idx in enumerate(tqdm(idx)):

        # Get patch (maybe pair)
        images = [x[i_idx] if x.shape[-1] == 3 else np.concatenate([x[i_idx], x[i_idx], x[i_idx]], axis=-1) for x in xs]
        image = np.concatenate(images, axis=1)
        label = y[i_idx]
        pred = preds[i_idx, ...]

        # Range
        image = image / 255.0

        # Encode
        if encode:
            key[i] = label
            output_path = join(output_dir, '%06d.png' % (i))
        else:
            if pred_path:
                if pred.dtype == 'uint8':
                    output_path = join(output_dir, '%d_%d_%06d.png' % (pred, label, i_idx))
                else:
                    output_path = join(output_dir, '%f_%d_%06d.png' % (pred, label, i_idx))
            else:
                output_path = join(output_dir, '%d_%06d.png' % (label, i_idx))

        # Save
        imsave_range(output_path, image.transpose((2, 0, 1)))

    # Key
    if encode:
        pd.DataFrame(key, index=[0]).T.to_excel(join(output_dir, 'key.xls'))

    # Clean
    del xs
    gc.collect()

#----------------------------------------------------------------------------------------------------

def imsave_range(output_path, image, range_min=0, range_max=1, channel_axis=0):
    """
    Saves arrays to image files similarly to what scipy.misc.imsave does. However, this function takes image range
    into account.

    Args:
        output_path: path to save the image.
        image: numpy array.
        range_min: low bound.
        range_max: high bound.
        channel_axis: color channel axis.

    """

    scipy.misc.toimage(image, cmin=range_min, cmax=range_max, channel_axis=channel_axis).save(output_path)

#----------------------------------------------------------------------------------------------------

def wsi_set_to_array(image_dir, mask_dir, output_dir, split_dict, draw_patches, max_patches_per_label,
                     image_pattern='*.mrxs', image_id_fn=None, mask_pattern='{image_id}_manual_labels.tif',
                     include_mask_patch=False, image_level=1, mask_level=5, patch_size=128, slide_output_dir=None,
                     cache_dir=None, selective_processing=False):

    if image_id_fn is None:
        image_id_fn = lambda x: splitext(basename(x))[0]

    image_paths = glob(join(image_dir, image_pattern))
    if slide_output_dir is None:
        slide_output_dir = join(output_dir, 'slides')

    if not exists(slide_output_dir):
        os.makedirs(slide_output_dir)
    if not exists(output_dir):
        os.makedirs(output_dir)

    # Collect all images that need to be processed
    images_to_be_processed = None
    if selective_processing:
        images_to_be_processed = []
        for _, values in split_dict.items():
            images_to_be_processed.extend(values)

    paths = {}
    for image_path in image_paths:

        image_id = image_id_fn(image_path)
        mask_path = join(mask_dir, mask_pattern.format(image_id=image_id))
        process_image = True if images_to_be_processed is None else image_id in images_to_be_processed

        if exists(mask_path) and process_image:

            print('Processing {image} ...'.format(image=image_path), flush=True)
            try:
                array_path, label_path = wsi_to_patches(
                    image_path=image_path,
                    mask_path=mask_path,
                    output_dir=slide_output_dir,
                    image_level=image_level,
                    mask_level=mask_level,
                    patch_size=patch_size,
                    max_patches_per_label=max_patches_per_label,
                    draw_patches=False,
                    include_mask_patch=include_mask_patch,
                    cache_dir=cache_dir
                )

                paths[image_id] = (array_path, label_path)

            except Exception as e:
                print('Exception: %s' % str(e), flush=True)

    # Concat images based on training/val/test split
    if split_dict is None:
        split_dict = {'training': [image_id_fn(path) for path in image_paths]}
    for split_name, split_list in split_dict.items():

        # Concat
        array = np.concatenate([np.load(paths[image_id][0]).astype('uint8') for image_id in split_list if image_id in list(paths.keys())], axis=0)
        labels = np.concatenate([np.load(paths[image_id][1]).astype('uint8') for image_id in split_list if image_id in list(paths.keys())], axis=0)

        # Randomize
        idxs = np.random.choice(len(labels), len(labels), replace=False)
        array = array[idxs, ...]
        labels = labels[idxs]

        # Store
        np.save(join(output_dir, '{tag}_x.npy'.format(tag=split_name)), array)
        np.save(join(output_dir, '{tag}_y.npy'.format(tag=split_name)), labels)

        # Dump
        if draw_patches:
            dump_patches(
                x_paths=[
                    join(output_dir, '{tag}_x.npy'.format(tag=split_name))
                ],
                y_path=join(output_dir, '{tag}_y.npy'.format(tag=split_name)),
                output_dir=join(output_dir, split_name + '_patches'),
                max_items=1000,
                encode=False
            )

#----------------------------------------------------------------------------------------------------

def wsi_to_patches(image_path, mask_path, output_dir, image_level, mask_level, patch_size, max_patches_per_label, draw_patches, include_mask_patch=False, cache_dir=None):

    # Check
    image_id = splitext(basename(image_path))[0]
    array_path = join(output_dir, '{tag}_x.npy'.format(tag=image_id))
    label_path = join(output_dir, '{tag}_y.npy'.format(tag=image_id))
    if not exists(array_path) and not exists(label_path):

        # Cache image and mask
        if cache_dir is not None:
            image_path = cache_file(image_path, cache_dir, overwrite=True)
            mask_path = cache_file(mask_path, cache_dir, overwrite=True)

        # Load image
        import multiresolutionimageinterface as mri
        image_reader = mri.MultiResolutionImageReader()
        image = image_reader.open(image_path)

        # Load mask
        mask_reader = mri.MultiResolutionImageReader()
        mask = mask_reader.open(mask_path)
        mask_shape = mask.getLevelDimensions(mask_level)
        image_mask_ratio = image.getLevelDimensions(0)[0] / mask.getLevelDimensions(0)[0]
        # assert image_mask_ratio == 1.0

        # Retrieve mask patch
        mask_tile = mask.getUCharPatch(
            0,
            0,
            mask_shape[0],
            mask_shape[1],
            mask_level
        )[:, :, 0]

        # Remove 255 values (filling).
        mask_tile[mask_tile == 255] = 0

        # For each label
        patches = []
        labels = []
        for label in np.unique(mask_tile):

            # Ignore background
            if label == 0:
                continue

            # Find locations
            locations = np.where(mask_tile.flatten() == label)[0]

            # Sample
            idxs = np.random.choice(len(locations), np.min([len(locations), max_patches_per_label]), replace=False)
            locations_sample = locations[idxs]
            locations_sample = np.unravel_index(locations_sample, mask_tile.shape)

            # Gather patches
            for y, x in zip(*locations_sample):
                patch = image.getUCharPatch(
                    int(x * image_mask_ratio * 2 ** mask_level - patch_size * 2 ** (image_level - 1)),
                    int(y * image_mask_ratio * 2 ** mask_level - patch_size * 2 ** (image_level - 1)),
                    patch_size,
                    patch_size,
                    image_level
                )
                if include_mask_patch:
                    mask_patch = mask.getUCharPatch(
                        int(x * 2 ** mask_level - patch_size * 2 ** (image_level - 1)),
                        int(y * 2 ** mask_level - patch_size * 2 ** (image_level - 1)),
                        patch_size,
                        patch_size,
                        image_level
                    )
                    cpm = mask_patch[64, 64, 0]
                    # cpm = mask_patch[0, 0, 0]
                    mask_patch = (mask_patch / np.max(np.unique(mask_tile)) * 255).astype('uint8')
                    patch = np.concatenate([patch, np.concatenate([mask_patch, mask_patch, mask_patch], axis=-1)], axis=1)
                    print('Central pixel mask: {cpm}, label: {l}'.format(cpm=cpm, l=label), flush=True)

                patches.append(patch)

            # Labels
            labels += list(np.ones(len(idxs)) * label)

        # Assemble
        patches = np.stack(patches, axis=0)
        labels = np.array(labels)

        # Randomize
        idxs = np.random.choice(len(labels), len(labels), replace=False)
        patches = patches[idxs, ...]
        labels = labels[idxs]

        # Store
        np.save(array_path, patches.astype('uint8'))
        np.save(label_path, labels.astype('uint8'))

        # Dump
        if draw_patches:
            dump_patches(
                x_paths=[
                    join(output_dir, '{tag}_x.npy'.format(tag=image_id))
                ],
                y_path=join(output_dir, '{tag}_y.npy'.format(tag=image_id)),
                output_dir=join(output_dir, image_id + '_patches'),
                max_items=1000,
                encode=False
            )

    return array_path, label_path


def map_dataset_labels(input_dir, output_dir, label_map, test_tag=None, draw_patches=True):

    # Locate paths
    x_paths = glob(join(input_dir, '*_x.npy'))

    for x_path in x_paths:

        # Names
        tag = basename(x_path).split('_')[0]
        y_path = join(dirname(x_path), tag + '_y.npy')
        if tag == 'test' and test_tag is not None:
            tag = test_tag
        x_path_output = join(output_dir, tag + '_x.npy')
        y_path_output = join(output_dir, tag + '_y.npy')

        # Copy X
        if not exists(x_path_output):
            shutil.copyfile(x_path, x_path_output)

        # Copy Y
        if not exists(y_path_output):

            # Map labels
            source_labels = np.load(y_path)
            labels = np.copy(source_labels)

            # Per label in the dataset
            for label in np.unique(source_labels):

                if label in list(label_map.keys()):
                    labels[source_labels == label] = label_map[label]
                else:
                    labels[source_labels == label] = 255

            # Store
            np.save(y_path_output, labels.astype('uint8'))

        # Dump
        if draw_patches:
            dump_patches(
                x_paths=[
                    x_path_output
                ],
                y_path=y_path_output,
                output_dir=join(output_dir, tag + '_patches'),
                max_items=1000,
                encode=False
            )

def sample_images_compare_std(data_dir, patch_tags, output_dir, n_samples):

    from model.data_augmentation import rgb_to_gray

    if not exists(output_dir):
        os.makedirs(output_dir)

    for patch_tag in patch_tags:

        # idxs = None
        for x_path in glob(join(data_dir, patch_tag if patch_tag != 'grayscale' else 'patches', 'test_*_x.npy')):

            # Read array
            test_tag = splitext(basename(x_path))[0]
            x = np.load(x_path)

            # Sample idxs
            # if idxs is None:
            #     idxs = np.random.choice(x.shape[0], n_samples, replace=False)
            idxs = np.arange(n_samples)

            # Sample images
            x = x[idxs, ...]

            # Grayscale
            if patch_tag == 'grayscale':
                x = rgb_to_gray(x)

            # Save images
            for i, image in enumerate(x):

                # Grayscale
                if image.shape[-1] == 1:
                    image = np.concatenate([image, image, image], axis=-1)

                # Range
                image = image / 255.0

                # Save
                output_path = join(output_dir, '{t}_{i}_{s}.png'.format(t=test_tag, i=idxs[i], s=patch_tag))
                imsave_range(output_path, image.transpose((2, 0, 1)))
                print(output_path)

            # Release memory
            del x
            gc.collect()


if __name__ == '__main__':

    # data_dir = r'Y:\projects\DataAugmentationComPat\data'
    # output_dir = r'Y:\projects\DataAugmentationComPat\data\stain_standardization\sample_patches'
    data_dir = r'/mnt/synology/pathology/projects/DataAugmentationComPat/data'
    output_dir = r'/mnt/synology/pathology/projects/DataAugmentationComPat/data/stain_standardization/sample_patches'

    organ_tags = ['rectum', 'lymph', 'mitosis', 'prostate']
    # patch_tags = ['patches', 'patches_std', 'patches_network-std-multi-organ', 'grayscale']
    patch_tags = ['patches_bugetal', 'patches_macenko']
    n_samples = 20

    for organ_tag in organ_tags:
        sample_images_compare_std(
            data_dir=join(data_dir, organ_tag),
            patch_tags=patch_tags,
            output_dir=join(output_dir, organ_tag),
            n_samples=n_samples
        )

