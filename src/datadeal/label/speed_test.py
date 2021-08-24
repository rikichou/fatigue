import sys
import os
import numpy as np
import argparse
import glob
import time
from pathlib import Path
from tqdm import tqdm
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_dir', type=str, help='root directory for the frames')
    parser.add_argument(
        '--level', type=int, default=4, help='dir level')
    parser.add_argument(
        '--ext', type=str, default='npy', help='dir level')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # read all npy
    st = time.time()
    print("Start to get all {} files".format(args.ext))
    files = glob.glob(args.src_dir + str(Path('/*' * args.level)) + '.' +
              args.ext)
    print("Glob get {} files, Time cost {} seconds".format(len(files), time.time()-st))

    # check parse time
    if args.ext == 'npy':
        for f in tqdm(files):
            info = np.load(f, allow_pickle=True).item()
    else:
        for f in tqdm(files):
            with open(f, 'rb') as fp:
                pkl_info = pickle.load(fp)

if __name__ == '__main__':
    main()






