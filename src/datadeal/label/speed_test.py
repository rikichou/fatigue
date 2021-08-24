import sys
import os
import numpy as np
import argparse
import glob
import time
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_dir', type=str, help='root directory for the frames')
    parser.add_argument(
        '--level', type=int, default=3, help='dir level')
    parser.add_argument(
        '--ext', type=str, default='npy', help='dir level')
    args = parser.parse_args()

    return args

def base_info_check(args, video_path, img_format, total_num):

    for i in range(total_num):
        img_path = os.path.join(video_path, img_format.format(i+args.start_idx))
        if not os.path.exists(img_path):
            print("ERROR! {} not in {}".format(img_format.format(i+args.start_idx), video_path))
            return False
    return True

no_facerect_count = 0
no_file_list = []
def get_valid_fatigue_idx(args, video_path, file_name, fatigue_idxs_str):
    global no_facerect_count
    global no_file_list

    # check if numpy file is exist
    file_path = os.path.join(video_path, file_name)
    if not os.path.exists(file_path):
        print("ERROR! {} not found in {}".format(file_name, video_path))
        no_file_list.append(file_path)
        return []

    # prepare fatigue index
    fatigue_idxs = [int(x) for x in fatigue_idxs_str]

    # prepare face rectangle idx map info
    rect_infos = np.load(file_path, allow_pickle=True).item()
    idx_rect_map = np.zeros(len(rect_infos), np.bool)
    for info in rect_infos:
        idx = int(info.split('.')[0].split('_')[1]) - 1
        if not rect_infos[info] is None:
            idx_rect_map[idx] = True

    # index check
    min_frames_before_fatigue = args.min_frames_before_fatigue
    valid_idxs = []
    for fat_end_idx in fatigue_idxs:
        fat_end_idx -= 1
        fat_start_idx = max(fat_end_idx - min_frames_before_fatigue + 1, 0)

        if idx_rect_map[fat_start_idx:fat_end_idx + 1].sum() == min_frames_before_fatigue:
            # valid idx
            valid_idxs.append(fat_end_idx + 1)

    return valid_idxs

def main():
    args = parse_args()

    # read all npy
    st = time.time()
    print("Start to get all {} files".format(args.ext))
    files = glob.glob(args.src_dir + str(Path('/*' * args.level)) + '.' +
              args.ext)
    print("Glob get {} files, Time cost {} seconds".format(len(files), time.time()-st))

    # check parse time
    for f in tqdm(files):
        info = np.load(f, allow_pickle=True).item()

if __name__ == '__main__':
    main()






