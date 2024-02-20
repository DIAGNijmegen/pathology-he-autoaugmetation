"""
This allows you to copy files using glob patterns.
"""

from os.path import basename, dirname, join, exists, splitext
import os
from tqdm import tqdm
import shutil
from glob import glob
import argparse
import numpy as np
import pandas as pd

#----------------------------------------------------------------------------------------------------

def cache_file(path, cache_dir, overwrite):
    """
    Copies a file to the cache directory if it is not already there. Copies auxiliary slide directory if needed.

    Args:
        path (str): path to file.

    Returns: path of the new file (same as input if there was a problem copying).

    """

    cache_path = path

    if cache_dir is not None:
        if not exists(cache_dir):
            os.makedirs(cache_dir)

        cache_path = join(cache_dir, basename(path))
        if not exists(cache_path) or overwrite:
            try:
                if exists(cache_path):
                    os.remove(cache_path)

                print('Caching file %s ...' % cache_path, flush=True)
                shutil.copyfile(path, cache_path)

                # Copy aux directory
                path_aux_dir = join(dirname(path), splitext(basename(path))[0])
                path_aux_dir_out = join(cache_dir, splitext(basename(path))[0])
                if exists(path_aux_dir):
                    if exists(path_aux_dir_out):
                        shutil.rmtree(path_aux_dir_out)

                    print('Caching aux folder %s ...' % path_aux_dir_out, flush=True)
                    shutil.copytree(path_aux_dir, path_aux_dir_out)

            except Exception as e:
                print('Failed to cache file %s. Using original location instead. Exception: %s' % (cache_path, str(e)), flush=True)
                cache_path = path

    return cache_path

#----------------------------------------------------------------------------------------------------

def copy_files(input_pattern, output_dir, overwrite):

    # List paths
    paths = glob(input_pattern)

    # Create output dir
    if not exists(output_dir):
        os.mkdir(output_dir)

    # Copy files
    for input_path in tqdm(paths):

        try:
            output_path = join(output_dir, basename(input_path))
            if not exists(output_path) or overwrite:
                shutil.copyfile(input_path, output_path)

        except Exception as e:
            print('Error copying {file}. Exception: {e}'.format(file=input_path, e=e))

#----------------------------------------------------------------------------------------------------

def collect_arguments():
    """
    Collect command line arguments.
    """

    # Configure argument parser.
    #
    argument_parser = argparse.ArgumentParser(description='Copy files using glob patterns.')

    argument_parser.add_argument('-i', '--input',     required=True,  type=str,            help='input image')
    argument_parser.add_argument('-o', '--output',    required=True,  type=str,            help='output image')
    argument_parser.add_argument('-w', '--overwrite', action='store_true',                 help='overwrite existing results')

    # Parse arguments.
    #
    arguments = vars(argument_parser.parse_args())

    # Collect arguments.
    #
    parsed_input_path = arguments['input']
    parsed_output_path = arguments['output']
    parsed_overwrite = arguments['overwrite']

    # Print parameters.
    #
    print(argument_parser.description)
    print('Input image: {input_path}'.format(input_path=parsed_input_path))
    print('Output image: {output_path}'.format(output_path=parsed_output_path))
    print('Overwrite existing results: {overwrite}'.format(overwrite=parsed_overwrite), flush=True)

    return parsed_input_path, parsed_output_path, parsed_overwrite

#----------------------------------------------------------------------------------------------------

def test_integrity_npy(pattern, output_path):

    data = []
    for path in tqdm(glob(pattern)):

        okay = True
        f = basename(path)
        e = ''

        try:
            array = np.load(path).astype('float32')
            s = array.shape
            m = 'File {f} has shape {s}'.format(f=path, s=s)

        except Exception as e:
            okay = False
            s = ''
            m = 'Failed {f} with exception: {e}'.format(f=path, e=e)

        print(m, flush=True)
        data.append({
            'file': f,
            'shape': str(s),
            'exception': str(e),
            'okay': str(okay)
        })

    pd.DataFrame(data).to_csv(output_path)

#----------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # parsed_input_path, parsed_output_path, parsed_overwrite = collect_arguments()
    # copy_files(parsed_input_path, parsed_output_path, parsed_overwrite)
    test_integrity_npy(
        pattern=r"/mnt/synology/pathology/projects/BreastCancerPredictionWSI/journal2/data/featurized_wsi/tupac16/fold_0/bigan/exp_id_1/all/*TE*.npy",
        output_path=r"/mnt/synology/pathology/projects/BreastCancerPredictionWSI/journal2/data/featurized_wsi/tupac16/fold_0/bigan/exp_id_1/integrityt16_test.csv"
    )