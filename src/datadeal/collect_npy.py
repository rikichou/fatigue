import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Pool

import shutil
import tqdm
import cv2
from PIL import Image
import mmcv

from pathlib import Path
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help='directory level of data')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='number of workers to build rawframes')
    parser.add_argument(
        '--ext',
        type=str,
        default='npy',
        choices=['avi', 'mp4', 'webm'],
        help='numpy file extensions')

    args = parser.parse_args()

    return args

args = parse_args()
if __name__ == '__main__':
    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    print('Reading npy from folder: ', args.src_dir)
    print('Extension of npy: ', args.ext)

    fullpath_list = glob.glob(args.src_dir + str(Path('/*' * args.level)) + '.' +
                              args.ext)
    done_fullpath_list = glob.glob(args.out_dir + str(Path('/*' * args.level)))
    print('Total number of npy found: ', len(fullpath_list))
    assert (args.level==4, "Error level : ".format(args.level))

    # copy data
    for idx,f in enumerate(fullpath_list):
        vid_prefix = os.path.join(*(f.rsplit(str(Path('/')), maxsplit=4)[-4:]))
        out_npy_path = os.path.join(args.out_dir, vid_prefix)
        out_npy_dir = os.path.dirname(out_npy_path)
        if not os.path.isdir(out_npy_dir):
            os.makedirs(out_npy_dir)
        shutil.copy(f, out_npy_path)
        if idx%1000==0:
            print("{}/{}".format(idx, len(fullpath_list)))



