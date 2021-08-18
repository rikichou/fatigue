import sys
import os
import numpy as np

DATA_ROOT_DIR = ''
LABEL_FILE_PATH = ''

start_idx = 1
img_format = '{:05}.jpg'

def base_info_check(root_dir, video_prefix, img_format, total_num):
    video_path = os.path.join(root_dir, video_prefix)

    for i in range(total_num):
        img_path = os.path.join(video_path, img_format.format(i+1))
        if not os.path.exists(img_path):
            print("ERROR! {} not in {}".format(img_format.format(i+1), video_path))
            return False
    return True


with open(LABEL_FILE_PATH, 'r') as fp:
    for line in fp:
        # video_prefix, total_num, label, indxs
        tmp_split = line.strip().split(',')
        line_split = tmp_split[:2]
        fatigue_idxs = tmp_split[3]

        # base info check
        video_prefix = line_split[0]
        total_num = int(line_split[1])
        video_path = os.path.join(DATA_ROOT_DIR, video_prefix)

        base_info_check(DATA_ROOT_DIR, video_prefix, img_format, total_num)

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





