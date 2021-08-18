import sys
import os
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames')
    parser.add_argument(
        'label_path', type=str, help='label path')
    parser.add_argument(
        '--start_idx', type=int, default=1, help='image frames start index')
    parser.add_argument(
        '--img_format', type=str, default='img_{:05}.jpg', help='image name format')
    args = parser.parse_args()

    return args

def base_info_check(args, root_dir, video_prefix, img_format, total_num):
    video_path = os.path.join(root_dir, video_prefix)

    for i in range(total_num):
        img_path = os.path.join(video_path, img_format.format(i+args.start_idx))
        if not os.path.exists(img_path):
            print("ERROR! {} not in {}".format(img_format.format(i+args.start_idx), video_path))
            return False
    return True

def main():
    args = parse_args()

    with open(args.label_path, 'r') as fp:
        for line in fp:
            # video_prefix, total_num, label, indxs
            tmp_split = line.strip().split(',')
            line_split = tmp_split[:2]
            fatigue_idxs = tmp_split[3]

            # base info check
            video_prefix = line_split[0]
            total_num = int(line_split[1])
            video_path = os.path.join(args.src_folder, video_prefix)

            base_info_check(args, args.src_folder, video_prefix, args.img_format, total_num)

            # # face rect check
            # facerect_path = os.path.join(video_path, 'facerect.npy')
            # if not os.path.exists(facerect_path):
            #     print("Warning! Face rectangel not found {}".format(video_path))
            #     continue
            #
            # rect_infos = np.load(facerect_path).item()
            # for info in rect_infos:
            #     if rect_infos[info] is None:
            #         print("Have no rectangle {}:{}".format(video_path, info))
            #
            # # image check
            # for idx in range(int(line_split[1])):
            #     img_path = os.path.join(video_path, img_format.format(idx))
            #     if not os.path.exists(img_path):
            #         print('Image not exist!!! {}'.format(img_path))


if __name__ == '__main__':
    main()






