''' Standardize patches to have similar stain appearance '''

from os.path import basename, dirname, join, exists, splitext
import os
from tqdm import tqdm
import numpy as np
from glob import glob
import shutil
from PIL import Image
import sys
import gc
import multiprocessing
from utils.data_handling import dump_patches

def normalizeStaining(img, saveFile=None, Io=240, alpha=1, beta=0.15):
    ''' Normalize staining appearence of H&E stained images

    Example use:
        see test.py

    Input:
        I: RGB input image XYC255
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # # read image
    # img = np.array(Image.open(img_path))

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    rimg = np.reshape(img.astype(np.float), (-1, 3))

    # calculate optical density
    OD = -np.log((rimg + 1) / Io)

    # remove transparent pixels
    ODhat = np.array([i for i in OD if not any(i < beta)])
    if len(ODhat) <= 0:
        raise Exception('Not enough pixels above the threshold.')

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]
    # C = np.linalg.lstsq(HE, Y)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    C2 = np.array([C[:, i] / maxC * maxCRef for i in range(C.shape[1])]).T

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    if saveFile is not None:
        Image.fromarray(Inorm).save(saveFile + '.png')
        Image.fromarray(H).save(saveFile + '_H.png')
        Image.fromarray(E).save(saveFile + '_E.png')

    return Inorm

def standardize_dataset(dataset_dir, organ_tag, dataset_tag='*_x.npy', workers=1, use_cache=False):

    # Paths
    output_dir = join(dataset_dir, organ_tag, 'patches_macenko')
    if not exists(output_dir):
        os.makedirs(output_dir)

    # Iterate
    x_pattern = join(dataset_dir, organ_tag, 'patches', dataset_tag)
    for x_path in glob(x_pattern):

        # Paths
        print('Standardizing {f} ...'.format(f=x_path), flush=True)
        filename = splitext(basename(x_path))[0]
        y_path = x_path[:-5] + 'y.npy'
        output_x_path = join(output_dir, filename[:-1] + 'x.npy')
        output_y_path = join(output_dir, filename[:-1] + 'y.npy')
        cache_dir = join(output_dir, filename[:-1] + '_cache') if use_cache else None

        if not exists(output_x_path):

            # Copy labels
            shutil.copyfile(y_path, output_y_path)

            # Standardize and plot patches
            standardize_array(
                input_path=x_path,
                output_path=output_x_path,
                workers=workers,
                cache_dir=cache_dir
            )
            gc.collect()


def standardize_array(input_path, output_path, workers, cache_dir=None):

    # Cache dir
    if cache_dir is not None and not exists(cache_dir):
        os.makedirs(cache_dir)

    # Read array
    x = np.load(input_path)

    if workers <= 1:

        # Apply to patches
        for i in tqdm(range(x.shape[0])):
            try:
                new_patch = normalizeStaining(x[i, ...])
                x[i, ...] = new_patch
            except Exception as e:
                print('Failed to process patch {i} with exception "{e}".'.format(i=i, e=e), flush=True)

    else:
        pool = multiprocessing.Pool(processes=workers)
        chunks = 10000
        jobs = [(x[i:i+chunks, ...], i, cache_dir) for i in range(0, x.shape[0], chunks)]
        x = pool.map(standardize_worker, jobs)

        if cache_dir is not None:
            x = [np.load(path) for path in glob(join(cache_dir, '*.npy'))]

        x = np.concatenate(x, axis=0)

    # Store
    # np.save(output_path, x.astype('uint8'))
    np.save(output_path, x)

def standardize_worker(args):

    x, j, cache_dir = args

    for i in tqdm(range(x.shape[0])):
        try:
            new_patch = normalizeStaining(x[i, ...])
            x[i, ...] = new_patch
        except Exception as e:
            print('Failed to process patch {i} with exception "{e}".'.format(i=i, e=e), flush=True)

    if cache_dir is not None:
        np.save(join(cache_dir, '{j}.npy'.format(j=j)), x)
        return None
    else:
        return x


if __name__ == '__main__':

    output_path = r"/mnt/synology/pathology/projects/DataAugmentationComPat/data/mitosis/patches_macenko/training_x.npy"
    source_path = r"/mnt/synology/pathology/projects/DataAugmentationComPat/data/mitosis/patches/training_x.npy"
    source_y_path = r"/mnt/synology/pathology/projects/DataAugmentationComPat/data/mitosis/patches_macenko/training_y.npy"
    plot_dir = r"/mnt/synology/pathology/projects/DataAugmentationComPat/data/mitosis/patches_macenko/training_patches"
    pattern = r"/mnt/synology/pathology/projects/DataAugmentationComPat/data/mitosis/patches_macenko/training__cache/{i}.npy"
    # x = []
    # paths = [pattern.format(i=i*10000) for i in range(52)]
    # for path in tqdm(paths):
    #     print(path, flush=True)
    #     x.append(np.load(path))
    # x = np.concatenate(x, axis=0)
    # np.save(output_path, x)

    dump_patches(
        x_paths=[
            source_path, output_path
        ],
        y_path=source_y_path,
        output_dir=plot_dir,
        max_items=500,
        encode=False
    )


    # standardize_dataset(
    #     dataset_dir=sys.argv[1],
    #     organ_tag=sys.argv[2],
    #     dataset_tag=sys.argv[3],
    #     workers=int(sys.argv[4]),
    #     use_cache=True if sys.argv[5] == 'True' else False
    # )
    # standardize_dataset(
    #     dataset_dir=r"C:\Users\david\Downloads\debug_macenko",
    #     organ_tag='prostate',
    #     workers=4
    # )

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
