''' Standardize patches to have similar stain appearance '''

from os.path import basename, dirname, join, exists, splitext
import os
from tqdm import tqdm
import numpy as np
import argparse
from glob import glob
import shutil
import time
import pandas as pd
from scipy.ndimage import imread
import scipy.misc
from PIL import Image
import sys
from subprocess import call
import gc

from utils.data_handling import dump_patches
from digitalpathology.image import imagewriter, imagereader
from utils.copy_files import cache_file


def apply_lut(patch, lut):

    ps = patch.shape # patch size is (rows, cols, channels)
    reshaped_patch = patch.reshape((ps[0]*ps[1], 3))
    normalized_patch = np.zeros((ps[0]*ps[1], 3))
    idxs = range(ps[0]*ps[1])
    Index = 256 * 256 * reshaped_patch[idxs,0] + 256 * reshaped_patch[idxs,1] + reshaped_patch[idxs,2]
    normalized_patch[idxs] = lut[Index.astype(int)]
    return normalized_patch.reshape(ps[0], ps[1], 3).astype('uint8')


def standardize_array(input_path, lut_path, output_path, plot_dir=None, y_path=None):

    # Read array
    x = np.load(input_path)

    # Reshape
    # x = x.transpose((0, 3, 1, 2))

    # Read LUT
    lut = Image.open(lut_path)
    lut_array = np.asarray(lut)[:, 0, :].astype('float32')

    # Apply to patches
    for i in tqdm(range(x.shape[0])):

        x[i, ...] = apply_lut(x[i, ...], lut_array)

    # Reshape
    # x = x.transpose((0, 2, 3, 1))

    # Store
    np.save(output_path, x.astype('uint8'))

    # Plot
    if plot_dir is not None:
        dump_patches(
            x_paths=[
                input_path, output_path
            ],
            y_path=y_path,
            output_dir=plot_dir,
            max_items=1000,
            encode=False
        )

#----------------------------------------------------------------------------------------------------

def create_wsi_from_patches(x_path, output_path, spacing, n_patches=5000):

    # Read data
    x = np.load(x_path)

    # Sample
    if n_patches is not None:
        x = x[np.random.choice(x.shape[0], np.min([n_patches, x.shape[0]])), ...]

    # Params
    n_patches = x.shape[0]
    patch_size = x.shape[1]
    n_rows = int(np.sqrt(n_patches) * 0.75)
    n_columns = n_patches // n_rows

    # Configure
    tile_size = patch_size
    image_writer = imagewriter.ImageWriter(
        image_path=output_path,
        shape=(n_columns * patch_size, n_rows * patch_size),
        spacing=spacing,
        dtype=np.uint8,
        coding='rgb',
        compression='jpeg',
        interpolation=None,
        tile_size=tile_size,
        jpeg_quality=None,
        empty_value=0,
        skip_empty=None
    )

    # Write
    counter = 0
    for row in tqdm(range(0, n_rows)):
        for col in range(0, n_columns):
            image_writer.write(tile=x[counter, ...].transpose((2, 0, 1)), row=col*tile_size, col=row*tile_size)
            counter += 1

            if counter >= n_patches:
                break
        if counter >= n_patches:
            break

    # Close
    image_writer.close()

#----------------------------------------------------------------------------------------------------

def change_tile_size(input_path, output_path, tile_size, copy=1):

    # Read
    image = imagereader.ImageReader(image_path=input_path, spacing_tolerance=0.25, input_channels=None)

    # Configure write
    image_shape = image.shapes[0]
    image_new_shape = (int(image.shapes[0][0]*copy), int(image.shapes[0][1]*copy))
    image_writer = imagewriter.ImageWriter(
        image_path=output_path,
        shape=image_new_shape,
        spacing=image.spacings[0],
        dtype=np.uint8,
        coding='rgb',
        compression=None,
        interpolation=None,
        tile_size=tile_size,
        jpeg_quality=None,
        empty_value=0,
        skip_empty=False
    )

    # Write
    for row in range(0, image_new_shape[0], tile_size):
        for col in range(0, image_new_shape[1], tile_size):
            image_patch = image.read(spacing=image.spacings[0], row=int(np.mod(row, image_shape[0])), col=int(np.mod(col, image_shape[1])), height=tile_size, width=tile_size)
            image_writer.write(tile=image_patch, row=row, col=col)

    image_writer.close()

#----------------------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------------------

def std_slide_network(input_path, output_path, model_path, tile_size=128):

    # Read
    image = imagereader.ImageReader(image_path=input_path, spacing_tolerance=0.25, input_channels=None)

    # Configure write
    image_shape = image.shapes[0]
    image_writer = imagewriter.ImageWriter(
        image_path=output_path,
        shape=image_shape,
        spacing=image.spacings[0],
        dtype=np.uint8,
        coding='rgb',
        compression=None,
        interpolation=None,
        tile_size=tile_size,
        jpeg_quality=None,
        empty_value=0,
        skip_empty=True
    )

    # Model
    from utils import dl
    model = dl.models.load_model(model_path)

    # Input independent of size
    x_input = dl.layers.Input((None, None, 3))
    x = model(x_input)
    model2 = dl.models.Model(inputs=x_input, outputs=x)

    # Write
    counter = 0
    padding = 16
    for row in tqdm(range(0, image_shape[0], tile_size)):
        for col in range(0, image_shape[1], tile_size):

            # Read patch
            image_patch = image.read(spacing=image.spacings[0], row=row-padding, col=col-padding, height=tile_size+padding*2, width=tile_size+padding*2)

            # Pad borders
            # image_patch = np.pad(image_patch, ((0, 0), (16, 16), (16, 16)), 'constant', constant_values=255)

            # Range
            image_patch = (image_patch / 255.0 - 0.5) * 2

            # Transpose
            image_patch = image_patch.transpose((1, 2, 0))[np.newaxis, ...]

            # Predict
            pred = model2.predict(image_patch)[0, ...].transpose(((2, 0, 1)))

            # Crop
            pred = pred[:, padding:-padding, padding:-padding]

            # Format
            pred = ((pred * 0.5 + 0.5) * 255.0).astype('uint8')

            # Save
            image_writer.write(tile=pred, row=row, col=col)

            counter += 1

    image_writer.close()

#----------------------------------------------------------------------------------------------------


#---------------------------------


def print_slide_properties(input1_path, input2_path):

    # Read
    image1 = imagereader.ImageReader(image_path=input1_path, spacing_tolerance=0.25, input_channels=None)
    image2 = imagereader.ImageReader(image_path=input2_path, spacing_tolerance=0.25, input_channels=None)

    # Properties
    properties = ['channels', 'close', 'coding', 'correct', 'downsamplings', 'dtype', 'hash', 'image', 'level', 'levels', 'path', 'read', 'shapes', 'spacings']

    for prop in properties:
        print('{p}: '.format(p=prop))
        print('\t{a}'.format(a=getattr(image1, prop)))
        print('\t{a}'.format(a=getattr(image2, prop)))


#----------------------------------------------------------------------------------------------------

def compute_lut(input_path, output_dir, number_of_samples=None):

    STD_EXE_PATH = r"C:\Science\Standardization_V1.4.0\SlideStandardization.exe"  # only Windows

    try:
        if output_dir is None:
            output_dir = dirname(input_path)

        output_path = join(output_dir, basename(input_path)[:-4] + '_LUT.tif')
        if not exists(output_path):
            call([STD_EXE_PATH, "-i", input_path, "-o", output_dir, "-n", str(number_of_samples), "-m", str(number_of_samples)])

    except Exception as e:
        print('Error compute LUT for image %s. Exception: %s' %(input_path, str(e)))


#----------------------------------------------------------------------------------------------------

def create_wsi_for_datasets(x_pattern, spacing_dict, output_dir=None):

    for x_path in glob(x_pattern):
        print(x_path, flush=True)
        test_tag = basename(x_path)[:-4].split('_')[1]

        if output_dir is None:
            output_dir = join(dirname(x_path), 'standardization')
            if not exists(output_dir):
                os.makedirs(output_dir)

        output_path = join(output_dir, basename(x_path)[:-4] + '.tif')
        if not exists(output_path):
            create_wsi_from_patches(
                x_path=x_path,
                output_path=output_path,
                spacing=spacing_dict[test_tag]
            )

#----------------------------------------------------------------------------------------------------

def sample_datasets(x_pattern, n_samples, output_dir):

    for x_path in glob(x_pattern):
        print(x_path, flush=True)

        # Read data
        x = np.load(x_path)

        # Sample
        if n_samples is not None:
            x = x[np.random.choice(x.shape[0], np.min([n_samples, x.shape[0]])), ...]

        # Save
        np.save(join(output_dir, basename(x_path)), x.astype('uint8'))

        del x
        gc.collect()


#----------------------------------------------------------------------------------------------------

def standardize_dataset(dataset_dir, lut_dir, organ_tag, internal_center_tag):

    # Paths
    output_dir = join(dataset_dir, organ_tag, 'patches_std')
    if not exists(output_dir):
        os.makedirs(output_dir)

    # Iterate
    x_pattern = join(dataset_dir, organ_tag, 'patches', '*_x.npy')
    for x_path in glob(x_pattern):

        # Paths
        print('Standardizing {f} ...'.format(f=x_path), flush=True)
        filename = splitext(basename(x_path))[0]
        y_path = x_path[:-5] + 'y.npy'
        purpose = filename.split('_')[0]
        output_x_path = join(output_dir, filename[:-1] + 'x.npy')
        output_y_path = join(output_dir, filename[:-1] + 'y.npy')

        if not exists(output_x_path):

            if purpose == 'training' or purpose == 'validation':
                lut_path = join(lut_dir, internal_center_tag + '_LUT.tif')
            else:
                lut_path = join(lut_dir, filename + '_LUT.tif')

            # Copy labels
            shutil.copyfile(y_path, output_y_path)

            # Standardize and plot patches
            standardize_array(
                input_path=x_path,
                lut_path=lut_path,
                output_path=output_x_path,
                plot_dir=join(output_dir, filename[:-1] + 'patches'),
                y_path=output_y_path
            )

            gc.collect()


#----------------------------------------------------------------------------------------------------

def reformat_slide(input_path, output_path, level=0, overwrite=True):

    from digitalpathology.processing.zoom import save_image_at_level

    save_image_at_level(
        image=input_path,
        output_path=output_path,
        level=level,
        pixel_spacing=None,
        spacing_tolerance=0.2,
        jpeg_quality=None,
        overwrite=overwrite,
        logger=None
    )

#----------------------------------------------------------------------------------------------------

def repeat_slide(input_path, output_path, repeat, empty_value=0, skip_empty=None, compression=None, tile_size=512):

    # Open images.
    #
    input_image = imagereader.ImageReader(image_path=input_path, spacing_tolerance=0.2, input_channels=None)

    # Calculate source level and target spacing and add missing spacing information.
    #
    processing_level = 0

    # Load patch and write it out.
    #
    image_shape_original = input_image.shapes[processing_level]

    # Repeat
    image_shape = (image_shape_original[0] * repeat, image_shape_original[1] * repeat)

    # Configure the image writer.
    #
    image_writer = imagewriter.ImageWriter(image_path=output_path,
                                              shape=image_shape,
                                              spacing=input_image.spacings[processing_level],
                                              dtype=input_image.dtype,
                                              coding=input_image.coding,
                                              compression=compression,
                                              interpolation=None,
                                              tile_size=tile_size,
                                              jpeg_quality=None,
                                              empty_value=empty_value,
                                              skip_empty=skip_empty)



    for row in tqdm(range(0, image_shape[0], tile_size)):
        for col in range(0, image_shape[1], tile_size):
            tile = input_image.read(
                spacing=input_image.spacings[processing_level],
                row=divmod(row, image_shape_original[0])[1],
                col=divmod(col, image_shape_original[1])[1],
                height=tile_size, width=tile_size
            )
            image_writer.write(tile=tile, row=row, col=col)

    # Finalize the output image.
    #
    image_writer.close()

#----------------------------------------------------------------------------------------------------

def compare_slides(input1_path, input2_path, output_path):

    from matplotlib import pyplot as plt

    image1 = imagereader.ImageReader(image_path=input1_path, spacing_tolerance=0.2, input_channels=None)
    image2 = imagereader.ImageReader(image_path=input2_path, spacing_tolerance=0.2, input_channels=None)

    data = {}
    data['filename'] = [basename(input1_path), basename(input2_path)]
    data['spacings'] = [image1.spacings, image2.spacings]
    data['shapes'] = [image1.shapes, image2.shapes]
    data['levels'] = [image1.levels, image2.levels]
    data['getspacing'] = [image1.image.getSpacing(), image2.image.getSpacing()]
    data['openslide.mpp-x'] = [image1.image.getProperty('openslide.mpp-x'), image2.image.getProperty('openslide.mpp-x')]

    f = lambda image: plt.hist(image.read(spacing=image.spacings[5], row=0, col=0, height=image.shapes[5][0], width=image.shapes[5][1]).flatten(), bins=20)

    data['hist'] = [[x for x in zip(*(f(image1)[:2]))], [x for x in zip(*(f(image2)[:2]))]]

    plt.hist(image1.read(spacing=image1.spacings[5], row=0, col=0, height=image1.shapes[5][0],
                         width=image1.shapes[5][1]).flatten(), bins=50, label='image1', alpha=0.5)
    plt.hist(image2.read(spacing=image2.spacings[5], row=0, col=0, height=image2.shapes[5][0],
                         width=image2.shapes[5][1]).flatten(), bins=50, label='image2', alpha=0.5)
    plt.legend()
    plt.show()

    df = pd.DataFrame(data).T
    print(df)

    df.to_csv(output_path)

def slide_to_image(input_path, output_path, level):

    from digitalpathology.processing.plainimage import save_mrimage_as_image

    save_mrimage_as_image(image=input_path,
                          output_path=output_path,
                          level=level,
                          pixel_spacing=None,
                          spacing_tolerance=0.25,
                          multiplier=1,
                          overwrite=True,
                          logger=None)

#----------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # x_pattern = r"Y:\projects\DataAugmentationComPat\sandbox\stain_standardization\sample_datasets\*_x.npy"
    # output_dir = r'C:\Users\david\Downloads\std_debug\create_wsi'
    # # x_pattern = r"/mnt/synology/pathology/projects/DataAugmentationComPat/data/*/patches/test_*_x.npy"
    # spacing_dict = {
    #     'tnbc': 0.25,
    #     'tupac': 0.25,
    #     'gertych': 0.5,
    #     'radboudumc1': 0.5,
    #     'radboudumc2': 0.5,
    #     'cwh': 0.25,
    #     'lpe': 0.25,
    #     'rh': 0.25,
    #     'rumc': 0.25,
    #     'umcu': 0.25,
    #     'heidelberg': 0.5,
    #     'labpon': 0.5,
    #     'radboudumc': 0.5
    # }
    #
    # create_wsi_for_datasets(x_pattern, spacing_dict, output_dir)

    # sample_datasets(
    #     x_pattern=x_pattern,
    #     n_samples=5000,
    #     output_dir=r'/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/stain_standardization/sample_datasets'
    # )

    # for wsi_path in glob(r"C:\Users\david\Downloads\std_debug\create_wsi\*.tif"):
    #     try:
    #         print(wsi_path)
    #         compute_lut(
    #             input_path=wsi_path,
    #             output_dir=r"Y:\projects\DataAugmentationComPat\sandbox\stain_standardization\sample_lut",
    #             number_of_samples=50000
    #         )
    #     except Exception as e:
    #         print(e)

    # standardize_array(
    #     input_path=r"Y:\projects\DataAugmentationComPat\sandbox\stain_standardization\sample_datasets\test_gertych_x.npy",
    #     lut_path=r"Y:\projects\DataAugmentationComPat\sandbox\stain_standardization\sample_lut\test_gertych_x_LUT.tif",
    #     output_path=r'Y:\projects\DataAugmentationComPat\sandbox\stain_standardization\std_datasets\test_gertych_x_std.npy',
    #     plot_dir=r'Y:\projects\DataAugmentationComPat\sandbox\stain_standardization\std_datasets\test_gertych_x_patches',
    # )

    # dataset_dir = r'Y:\projects\DataAugmentationComPat\data'
    # dataset_dir = r'/mnt/synology/pathology/projects/DataAugmentationComPat/data'
    # lut_dir = r'Y:\projects\DataAugmentationComPat\data\stain_standardization\sample_lut'
    # lut_dir = r'/mnt/synology/pathology/projects/DataAugmentationComPat/data/stain_standardization/sample_lut'

    # --------------------

    organ_internal_center_dict = {'rectum': 'test_radboudumc_x', 'lymph': 'test_rumc_x', 'mitosis': 'test_tnbc_x', 'prostate': 'test_radboudumc1_x'}

    organ = sys.argv[3]
    standardize_dataset(
        dataset_dir=sys.argv[1],
        lut_dir=sys.argv[2],
        organ_tag=organ,
        internal_center_tag=organ_internal_center_dict[organ]
    )

    #
    # ---------------------

    # reformat_slide(
    #     input_path=r"Y:\projects\DataAugmentationComPat\data\stain_standardization\sample_wsi\formats\test_cwh_x.tif",
    #     output_path=r"Y:\projects\DataAugmentationComPat\data\stain_standardization\sample_wsi\formats\test_cwh_x_reformat.tif",
    #     level=0
    # )
    # for i in range(3):
    #     reformat_slide(
    #         input_path=r"C:\Users\david\Downloads\stain_decomposition\normal_001.tif",
    #         output_path=r"C:\Users\david\Downloads\stain_decomposition\normal_001_reformat_lvl_%d.tif" % i,
    #         level=i,
    #         overwrite=True
    #     )

    # tile_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    #     # repeat = 1
    #     # empty_values = [0]
    #     # skip_empty = False
    #     # compressions = ['jpeg']
    #     #
    #     # for tile_size in tile_sizes:
    #     #     for empty_value in empty_values:
    #     #         for compression in compressions:
    #     #             repeat_slide(
    #     #                 # input_path=r"C:\Users\david\Downloads\stain_decomposition\test_cwh_x.tif",
    #     #                 input_path=r"C:\Users\david\Downloads\stain_decomposition\normal_001_reformat_lvl_3.tif",
    #     #                 # output_path=r"C:\Users\david\Downloads\stain_decomposition\test_cwh_x_repeat_%d_skip-False.tif" % r,
    #     #                 # output_path=r"C:\Users\david\Downloads\stain_decomposition\normal_001_reformat_lvl_1_repeat-{r}_empty-{e}_skip-{s}_compression-{c}.tif".format(r=repeat, e=empty_value, s=skip_empty, c=compression),
    #     #                 # output_path=r"Y:\projects\DataAugmentationComPat\data\stain_standardization\sample_wsi\formats\normal_001_repeat-{r}_empty-{e}_skip-{s}_compression-{c}_tile={t}.tif".format(r=repeat, e=empty_value, s=skip_empty, c=compression, t=tile_size),
    #     #                 output_path=r"C:\Users\david\Downloads\stain_decomposition\normal_001_reformat_lvl_3_repeat-{r}_empty-{e}_skip-{s}_compression-{c}_tile-{t}.tif".format(r=repeat, e=empty_value, s=skip_empty, c=compression, t=tile_size),
    #     #                 repeat=repeat,
    #     #                 empty_value=empty_value,
    #     #                 skip_empty=skip_empty,
    #     #                 compression=compression
    #     #             )

    # compare_slides(
    #     input1_path=r"C:\Users\david\Downloads\stain_decomposition\normal_001.tif",
    #     input2_path=r"Y:\projects\DataAugmentationComPat\data\stain_standardization\sample_wsi\formats\normal_001_repeat-1_empty-0_skip-False_compression-jpeg.tif",
    #     output_path=r"C:\Users\david\Downloads\stain_decomposition\comparison.csv"
    # )

    # level = 1
    # slide_to_image(
    #     input_path=r"C:\Users\david\Downloads\stain_decomposition\normal_001.tif",
    #     output_path=r"C:\Users\david\Downloads\stain_decomposition\normal_001_level-{l}.jpg".format(l=level),
    #     level=level
    # )


    # change_tile_size(
    #     input_path=r"C:\Users\david\Downloads\std_debug\stain_decomposition\test_cwh_x.tif",
    #     # input_path=r"Y:\projects\DataAugmentationComPat\data\stain_standardization\sample_wsi\formats\normal_001.tif",
    #     output_path=r"C:\Users\david\Downloads\std_debug\stain_decomposition\formats\test_cwh_x_none-none-512-none-0-8x.tif",
    #     tile_size=512,
    #     copy=8
    # )

    # for padding in [16, 32, 48, 64, 80, 96, 112, 128]:
    #
    #     try:
    #
    #         cache_dir = r'/home/user/data/data_augmentation/std-slide/cache'
    #         image_path = r"/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/standardizer/std-slide/TUPAC-TE-010.svs"
    #         output_path = r'/home/user/data/data_augmentation/std-slide/cache/TUPAC-TE-010_std_padding_{padding}.tif'.format(padding=padding)
    #         output_path_remote = r'/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/standardizer/std-slide/TUPAC-TE-010_std_{padding}.tif'.format(padding=padding)
    #         model_path = r"/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/standardizer/std-slide/checkpoint_paper.h5"
    #         std_slide_network_batch(
    #             input_path=cache_file(image_path, cache_dir, overwrite=True),
    #             output_path=output_path,
    #             model_path=model_path,
    #             tile_size=128,
    #             batch_size=128,
    #             padding=padding
    #         )
    #         shutil.copyfile(output_path, output_path_remote)
    #
    #     except Exception as e:
    #         print(e, flush=True)
    #         print('Exception happened.', flush=True)

    # tile_size = 2048-128
    # cache_dir = r'/home/user/data/data_augmentation/std-slide/cache'
    # image_path = r"/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/standardizer/std-slide/TUPAC-TE-010.svs"
    # output_path = r'/home/user/data/data_augmentation/std-slide/cache/TUPAC-TE-010_std_extended3_1_{ts}.tif'.format(ts=tile_size)
    # output_path_remote = r'/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/standardizer/std-slide/TUPAC-TE-010_std_extended3_1_{ts}.tif'.format(ts=tile_size)
    # # model_path = r"/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/standardizer/std-slide/checkpoint_paper.h5"
    # model_path = r"/mnt/synology/pathology/projects/DataAugmentationComPat/sandbox/standardizer/std-slide/checkpoint_extended3_1.h5"
    # std_slide_network_batch(
    #     input_path=cache_file(image_path, cache_dir, overwrite=True),
    #     output_path=output_path,
    #     model_path=model_path,
    #     tile_size=tile_size,
    #     batch_size=2,
    #     padding=64
    # )
    # # shutil.copyfile(output_path, output_path_remote)

    # std_slide_network_batch(
    #     input_path=r"C:\Users\david\Downloads\std_debug\network-std\TUPAC-TE-010.svs",
    #     output_path=r"C:\Users\david\Downloads\std_debug\network-std\TUPAC-TE-010_x_std_padding.tif",
    #     model_path=r"C:\Users\david\Downloads\std_debug\network-std\checkpoint_paper.h5",
    #     tile_size=128,
    #     batch_size=2,
    #     padding=128
    # )

    # print_slide_properties(
    #     input1_path=r"Y:\projects\DataAugmentationComPat\data\stain_standardization\sample_wsi\formats\normal_001.tif",
    #     input2_path=r"Y:\projects\DataAugmentationComPat\data\stain_standardization\sample_wsi\formats\test_cwh_x.tif",
    # )

    # reformat_slide(
    #     input_path=r"C:\Users\david\Downloads\pilot_neuro\3.tiff",
    #     output_path=r"C:\Users\david\Downloads\pilot_neuro\3_reformat.tif",
    #     level=0,
    #     overwrite=True
    # )
