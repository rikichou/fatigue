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
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
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
    print('Reading npy from folder: ', args.src_dir)

    fullpath_list = glob.glob(args.src_dir + str(Path('/*' * args.level)))
    print('Total number of files: ', len(fullpath_list))


    # with Pool(args.num_worker) as pool:
    #     r = list(tqdm.tqdm(pool.imap(
    #         deal,
    #         zip(fullpath_list, range(len(fullpath_list))))))

    # copy data
    for idx,f in enumerate(fullpath_list):
        name = os.path.basename(f)
        if ord(name[0]) == 9560 and ord(name[1]) == 9524:
            # rename
            new_name = '_'+name[2:]
            os.popen('mv {} {}'.format(f, os.path.join(os.path.dirname(f), new_name))).readlines()