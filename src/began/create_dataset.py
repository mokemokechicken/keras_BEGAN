# coding: utf8
"""
指定された64x64の画像ファイル群をDatasetとして変換する。

-> ndarray: (N, 64, 64, 3)
"""
import pickle
import sys
from glob import glob

from PIL import Image
import numpy as np

from began.config import BEGANConfig


def main(image_dir):
    config = BEGANConfig()
    convert_images_to_dataset(config, image_dir)


def convert_images_to_dataset(config: BEGANConfig, image_dir: list):
    files = glob("%s/*.jpg" % image_dir)  # Images must be resize to 64x64
    N = len(files)

    array_list = []
    for i in range(N):
        filename = files[i]
        im = Image.open(filename)  # type: Image.Image
        im_array = np.array(im, dtype='uint8')
        if im_array.ndim == 2:  # skip gray-scale image
            continue
        im_array = np.expand_dims(im_array, axis=0)
        array_list.append(im_array)
    seg_dataset = np.concatenate(array_list, axis=0)
    with open(config.dataset_filename, "wb") as f:
        pickle.dump(seg_dataset, f)


if __name__ == '__main__':
    main(sys.argv[1])
