import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Pool
import random

import tqdm
import cv2
from PIL import Image
import mmcv

from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory, if --random is set, src_dir is parent dir of video dir')
    parser.add_argument('out_dir', type=str, help='output frames directory')
    parser.add_argument(
        '--random',
        action='store_true',
        default=False,
        help='whether to random choose a video dir')
    args = parser.parse_args()

    return args


args = parse_args()
if __name__ == '__main__':
    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    if args.random:
        fs = os.listdir(args.src_dir)
        video_dir = os.path.join(args.src_dir, random.choice(fs))
    else:
        video_dir = args.src_dir
    print("Video dir : ", video_dir)
    label_file_path = os.path.join(video_dir, 'facerect.npy')
    if not os.path.exists(label_file_path):
        print("Have no facerect.py found in ", video_dir)
    infos_dict = np.load(label_file_path, allow_pickle=True).item()

    for item in infos_dict:
        sx,sy,ex,ey = infos_dict[item]
        img_path = os.path.join(video_dir, item)
        #image = cv2.imread(img_path)
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)

        cv2.rectangle(image, (sx,sy), (ex,ey), (255,0,0), 1)
        cv2.imwrite(os.path.join(args.out_dir, item), image)