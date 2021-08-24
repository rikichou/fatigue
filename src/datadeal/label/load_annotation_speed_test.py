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
        'ann_file', type=str, help='dir level')
    parser.add_argument(
        'data_prefix', type=str, default='/zhourui/workspace/pro/fatigue/data/rawframes/valid', help='dir level')
    parser.add_argument(
        'min_frames_before_fatigue', type=int, default=48, help='dir level')

    args = parser.parse_args()

    return args

def get_valid_fatigue_idx(rect_infos, min_frames_before_fatigue, fatigue_idxs_str):
    global no_facerect_count
    global no_file_list

    # prepare fatigue index
    fatigue_idxs = [int(x) for x in fatigue_idxs_str]

    # prepare face rectangle idx map info

    idx_rect_map = np.zeros(len(rect_infos), np.bool)
    for info in rect_infos:
        idx = int(info.split('.')[0].split('_')[1]) - 1
        if not rect_infos[info] is None:
            idx_rect_map[idx] = True

    # index check
    min_frames_before_fatigue = min_frames_before_fatigue
    valid_idxs = []
    for fat_end_idx in fatigue_idxs:
        fat_end_idx -= 1
        fat_start_idx = max(fat_end_idx - min_frames_before_fatigue + 1, 0)

        if idx_rect_map[fat_start_idx:fat_end_idx + 1].sum() == min_frames_before_fatigue:
            # valid idx
            valid_idxs.append(fat_end_idx + 1)

def load_annotations(args):
    video_infos = []
    print("Start to Parsing label file ", args.ann_file)

    # statistics info
    statistics_info = {}
    statistics_info['total'] = 0
    statistics_info['invalid'] = 0

    statistics_info[0] = {}
    statistics_info[1] = {}
    statistics_info[0]['num'] = 0
    statistics_info[0]['invalid'] = 0
    statistics_info[0]['clips'] = 0
    statistics_info[1]['num'] = 0
    statistics_info[1]['invalid'] = 0
    statistics_info[1]['clips'] = 0

    time_cose_ana = {}
    time_cose_ana['t1-t2'] = 0
    time_cose_ana['t2-t3'] = 0
    time_cose_ana['t3-t4'] = 0
    time_cose_ana['t4-t5'] = 0

    with open(args.ann_file, 'r') as fin:
        for idx,line in enumerate(fin):
            t1 = time.time()
            # video_prefix, total_frame_num, label, fatigue indexes
            tmp_split = line.strip().split(',')
            line_split = tmp_split[:3]
            fatigue_idxs_str = tmp_split[3:]

            video_prefix = line_split[0]
            video_prefix = video_prefix.replace('\\', '/')
            total_num = int(line_split[1])  # have no use
            fat_label = int(line_split[2])
            video_path = os.path.join(args.data_prefix, video_prefix)

            # statistics
            statistics_info['total'] += 1
            statistics_info[fat_label]['num'] += 1

            # video_path = Path(self.data_prefix, video_prefix)
            # get face_rectangle_infos
            t2 = time.time()
            facerect_file_path = os.path.join(video_path, 'facerect.npy')
            if not os.path.exists(facerect_file_path):
                statistics_info['invalid'] += 1
                statistics_info[fat_label]['invalid'] += 1
                print("!! Can not found ", facerect_file_path)
                continue
            rect_infos = np.load(facerect_file_path, allow_pickle=True).item()
            t3 = time.time()
            fat_idxs = get_valid_fatigue_idx(rect_infos, args.min_frames_before_fatigue, fatigue_idxs_str)
            if len(fat_idxs) < 1:
                statistics_info['invalid'] += 1
                statistics_info[fat_label]['invalid'] += 1
                continue
            t4 = time.time()
            # get each fatigue to video info
            infos = []
            for fat_end_idx in fat_idxs:
                video_info = {}
                video_info['facerect_infos'] = rect_infos
                video_info['fat_idxs'] = fat_idxs
                # idx for frame_dir
                frame_dir = video_prefix
                if args.data_prefix is not None:
                    frame_dir = os.path.join(args.data_prefix, frame_dir)
                video_info['frame_dir'] = frame_dir
                # get start_index
                video_info['start_index'] = fat_end_idx - args.min_frames_before_fatigue + 1
                video_info['total_frames'] = args.min_frames_before_fatigue

                # idx for label[s]
                video_info['label'] = fat_label
                infos.append(video_info)
            t5 = time.time()

            time_cose_ana['t1-t2'] += t2 - t1
            time_cose_ana['t2-t3'] += t3 - t2
            time_cose_ana['t3-t4'] += t4 - t3
            time_cose_ana['t4-t5'] += t5 - t4

            video_infos.append(infos)
            # statistics
            statistics_info[fat_label]['clips'] += len(fat_idxs)

            if idx%200 == 0:
                print(time_cose_ana)

    print(
        "Total {}\nInvalid {}\n\nFatigue_close {}\nInvalid {}\nValid {}\nClips {}, Clips_per_Video {}\n\nFatigue_look_down {}\nInvalid {}\nValid {}\nClips {}, Clips_per_Video {}".format(
            statistics_info['total'], statistics_info['invalid'],
            statistics_info[1]['num'], statistics_info[1]['invalid'],
            statistics_info[1]['num'] - statistics_info[1]['invalid'], statistics_info[1]['clips'],
            statistics_info[1]['clips'] / max(1, statistics_info[1]['num'] - statistics_info[1]['invalid']),
            statistics_info[0]['num'], statistics_info[0]['invalid'],
            statistics_info[0]['num'] - statistics_info[0]['invalid'], statistics_info[0]['clips'],
            statistics_info[0]['clips'] / max(1, statistics_info[0]['num'] - statistics_info[0]['invalid']),
        ))

    print(time_cose_ana)

    return video_infos

def main():
    args = parse_args()

    load_annotations(args)

if __name__ == '__main__':
    main()