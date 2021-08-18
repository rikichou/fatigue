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
        '--start_idx', type=int, default=1, help='image frames start index')
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
def face_rect_check(args, video_path, file_name):
    global no_facerect_count
    global no_file_list

    file_path = os.path.join(video_path, file_name)
    if not os.path.exists(file_path):
        print("ERROR! {} not found in {}".format(file_name, video_path))
        no_file_list.append(file_path)
        return False
    rect_infos = np.load(file_path, allow_pickle=True).item()

    for info in rect_infos:
        if rect_infos[info] is None:
            print("Have no face in {}:{}".format(video_path, info))
            no_facerect_count += 1
    return True

def main():
    args = parse_args()

    with open(args.label_path, 'r') as fp:
        for line in fp:
            # video_prefix, total_num, label, indxs
            tmp_split = line.strip().split(',')
            line_split = tmp_split[:2]
            fatigue_idxs = tmp_split[3]

            # base info get
            video_prefix = line_split[0]
            total_num = int(line_split[1])
            video_path = os.path.join(args.src_folder, video_prefix)

            # 1, base info check (check if image missing)
            # base_info_check(args, video_path, args.img_format, total_num)

            # 2, face rectange check (check if the have no face or face rectangle in an image)
            face_rect_check(args, video_path, args.facerect_filename)
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
    print("{} videos have no face bbox info, {} images have no face!".format(len(no_file_list), no_facerect_count))
    print("Fllowing videos have no face bbox!")
    for f in no_file_list:
        print(str(Path(f)))

if __name__ == '__main__':
    main()






