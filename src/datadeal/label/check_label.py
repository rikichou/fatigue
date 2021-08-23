import sys
import os
import numpy as np
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames')
    parser.add_argument(
        'label_path', type=str, help='label path')
    parser.add_argument(
        'message_save_file_path', type=str, help='label path')
    parser.add_argument(
        '--start_idx', type=int, default=1, help='image frames start index')
    parser.add_argument(
        '--min_frames_before_fatigue', type=int, default=32, help='min frames before fatigue warning')
    parser.add_argument(
        '--img_format', type=str, default='img_{:05}.jpg', help='image name format')
    parser.add_argument(
        '--facerect_filename', type=str, default='facerect.npy', help='numpy file include face bbox informations')
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
    fatigue_idxs = [int(x) for x in fatigue_idxs_str.strip().split(',')]

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

    invalid_info = []
    with open(args.label_path, 'r') as fp:
        for line in fp:
            # video_prefix, total_num, label, indxs
            tmp_split = line.strip().split(',')
            line_split = tmp_split[:2]
            fatigue_idxs_str = tmp_split[3]

            # base info get
            video_prefix = line_split[0]
            total_num = int(line_split[1])
            label = int(line_split[1])
            video_path = os.path.join(args.src_folder, video_prefix)

            # 1, base info check (check if image missing)
            # base_info_check(args, video_path, args.img_format, total_num)

            # 2, face rectange check (check if the have no face or face rectangle in an image)
            fat_idxs = get_valid_fatigue_idx(args, video_path, args.facerect_filename, fatigue_idxs_str)
            if len(fat_idxs) < 1:
                #print("{} : {}".format(fatigue_idxs_str, video_path))
                invalid_info.append("{} : {}\n".format(fatigue_idxs_str, video_path))

                # statistics
                statistics_info['invalid'] += 1
                statistics_info[label]['invalid'] += 1

            # statistics
            statistics_info['total'] += 1
            statistics_info[label]['num'] += 1
            statistics_info[label]['clips'] += len(fat_idxs)

        print("Total {}\nInvalid {}\n\nFatigue_close {}\nInvalid {}\nValid {}\nClips {}, Clips_per_Video {}\n\nFatigue_look_down {}\nInvalid {}\nValid {}\nClips {}, Clips_per_Video {}".format(
            statistics_info['total'], statistics_info['invalid'],
            statistics_info[1]['num'], statistics_info[1]['invalid'], statistics_info[1]['num']-statistics_info[1]['invalid'], statistics_info[1]['clips'], statistics_info[1]['clips']/(statistics_info[1]['num']-statistics_info[1]['invalid']),
            statistics_info[0]['num'], statistics_info[0]['invalid'], statistics_info[0]['num'] - statistics_info[0]['invalid'], statistics_info[0]['clips'], statistics_info[0]['clips']/(statistics_info[0]['num']-statistics_info[0]['invalid']),
        ))
        with open(args.message_save_file_path, 'w') as fp:
            fp.writelines(invalid_info)

if __name__ == '__main__':
    main()






