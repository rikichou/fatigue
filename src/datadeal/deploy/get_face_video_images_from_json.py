import os
import sys
import json
import random
import numpy as np

import cv2

ann_file = '/zhourui/workspace/pro/fatigue/data/clean/fatigue_anns/20210824_fatigue_pl_less_than_50_fatigue_full_info_all_path.json'
video_data_prefix = '/zhourui/workspace/pro/fatigue/data/rawframes/new_clean/fatigue_clips'
facerect_data_prefix = '/zhourui/workspace/pro/fatigue/data/clean/fatigue_info_from_yolov5'
out_root_dir = '/zhourui/workspace/pro/fatigue/data/lianyong/face_video_images'

if not os.path.exists(out_root_dir):
    os.makedirs(out_root_dir)

def get_facerects(self, filepath):
    facerect_infos = {}
    with open(filepath, 'r') as fp:
        anns = json.load(fp)
        for imgname in anns:
            facerect_infos[imgname] = anns[imgname]['bbox']
    return facerect_infos

def get_valid_fatigue_idx(self, rect_infos, min_frames_before_fatigue, fatigue_idxs, video_dir, max_frames=500):
    global no_facerect_count
    global no_file_list

    # prepare face rectangle idx map info
    idx_rect_map = np.zeros(max_frames+1, np.bool)
    for info in rect_infos:
        # just ignore index 0, both images and fatigue index are start with index 1
        idx = int(info.split('.')[0].split('_')[1])
        if not rect_infos[info] is None and os.path.exists(os.path.join(video_dir, info)):
            idx_rect_map[idx] = True

    # index check
    min_frames_before_fatigue = min_frames_before_fatigue
    valid_idxs = []
    for fat_end_idx in fatigue_idxs:
        fat_start_idx = max(fat_end_idx - min_frames_before_fatigue + 1, 1)
        # border control
        if idx_rect_map[fat_start_idx:fat_end_idx + 1].sum() == min_frames_before_fatigue:
            # valid idx
            valid_idxs.append(fat_end_idx)

    return valid_idxs

def load_annotations(ann_file, video_data_prefix, out_root_dir, data_phase='train', item_num=200, min_frames_before_fatigue=64):
    
    print("Start to Parsing label file ", ann_file)
    with open(ann_file, 'r') as fp:
        anns = json.load(fp)
        anns_keys = anns.keys()
        random.shuffle(anns_keys)

        count_map = {'fatigue_close': 0, 'fatigue_dahaqian': 0, 'fatigue_look_down': 0, 'fatigue_squint':0, 'smoking': 0, 'calling': 0}

        # select anns
        print("Start to select anns!")
        selected_anns = {}
        for idx,vname in enumerate(anns_keys):
            vinfo = anns[vname]

            # select video
            if count_map[vinfo['label']] < item_num:
                selected_anns[vname] = anns[vname]
                # video path
                video_path = os.path.join(video_data_prefix, vname)
                # total frames
                total_frames = vinfo['frames_avi']
                # fatigue_idxs
                fatigue_idxs = vinfo['fatigue_warning_idx']
                # video face rectangle
                facerect_path = os.path.join(facerect_data_prefix, vname + '.json')
                if not os.path.exists(facerect_path):
                    continue
                rect_infos = get_facerects(facerect_path)

                # get valid fatigue index according to facerect infos and fatigue index
                fat_idxs = get_valid_fatigue_idx(rect_infos, min_frames_before_fatigue, fatigue_idxs, video_path, max_frames=total_frames)
                if len(fat_idxs) < 1:
                    continue

                # dump face images to dir
                fat_end_idx = random.choice(fat_idxs)
                fat_start_idx = fat_end_idx - min_frames_before_fatigue + 2
                imgs_idxs = list(range(fat_start_idx, fat_end_idx+1, 2))

                for img_idx in imgs_idxs:
                    img_name = 'img_{:05d}.jpg'.format(img_idx)
                    img_path = os.path.join(video_path, img_idx)

                    out_dir = os.path.join(out_root_dir, vname)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    dst_img_path = os.path.join(out_dir, img_name)

                    # get face images
                    image = cv2.imread(img_path)
                    rect = rect_infos[img_name]

                    sx, sy, ex, ey = rect
                    h, w, c = image.shape

                    sx = int(max(0, sx))
                    sy = int(max(0, sy))
                    ex = int(min(w - 1, ex))
                    ey = int(min(h - 1, ey))

                    face_image = image[sy:ey, sx:ex, :]

                    # write
                    cv2.imwrite(dst_img_path, face_image)

                # count
                count_map[vinfo['label']] += 1
            
            if idx%100 == 0:
                print("{}, {}".format(idx, count_map))

            # check valid
            if len(count_map)*item_num <= sum(count_map.values()):
                break

load_annotations(ann_file, video_data_prefix, out_root_dir)