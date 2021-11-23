import sys
import os

from mmpose.apis import (inference_bottom_up_pose_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
import glob
import argparse
from multiprocessing import Process, Lock, Value
import cv2
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_facerect_json_dir', type=str, help='root directory for the frames')
    parser.add_argument(
        'src_rawframes_dir', type=str, default='facerect.npy', help='save name')
    parser.add_argument(
        'out_json_dir', type=str, default='facerect.npy', help='save name')
    parser.add_argument(
        '--level',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='directory level of data')
    parser.add_argument(
        '--num_worker',
        type=int,
        default=10,
        help='num workers to preprocess')
    parser.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help='whether to use cpu')

    args = parser.parse_args()
    return args

import json
def get_facerects(filepath):
    facerect_infos = {}
    with open(filepath, 'r') as fp:
        anns = json.load(fp)
        for imgname in anns:
            facerect_infos[imgname] = anns[imgname]['bbox']
    return facerect_infos

def process_paths(paths, args, lock, counter, total_length):
    # init pose model
    pose_model = init_pose_model(
        r'/zhourui/workspace/pro/source/mmpose/configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/300w/hrnetv2_w18_300w_256x256.py',
        r'/zhourui/workspace/pro/source/mmpose/pretrained/face/hrnetv2_w18_300w_256x256-eea53406_20211019.pth',
        device='cuda')
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))

    for video_facerect_json in paths:
        #  get json info
        facerects = get_facerects(video_facerect_json)
        vname = os.path.splitext(os.path.basename(video_facerect_json))[0]
        vdir = os.path.join(args.src_rawframes_dir, vname)

        images_rect_dict = {}
        for img in facerects:
            # get image path
            img_path = os.path.join(vdir, img)
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
            if image is None:
                print("!!!!!!!!!!!!!!!!!!!! Failed to read image ", img_path)
                assert (False)

            # get facerect
            facerect = facerects[img]

            imgname = os.path.basename(img_path)
            images_rect_dict[imgname] = {}

            sx, sy, ex, ey = facerect
            # get face rectangle, have no face rectangle, set None
            images_rect_dict[imgname]['facerect'] = [sx,sy,ex,ey]

            # test a single image, with a list of bboxes.
            image = image[sy:ey, sx:ex]
            pose_results, _ = inference_top_down_pose_model(
                pose_model,
                image,
                None,
                format='xywh',
                dataset_info=dataset_info)
            points = pose_results[0]['keypoints']
            images_rect_dict[imgname]['mmpose_kpts'] = points

        #np.save(os.path.join(path, args.out_name), images_rect_dict)
        label_name = os.path.join(args.out_json_dir, os.path.basename(video_facerect_json))
        with open(label_name, 'w') as f:
            json.dump(images_rect_dict, f)

        # counter
        lock.acquire()
        try:
            # p_bar.update(1)
            counter.value += 1
            if counter.value % 50 == 0:
                print(f"{counter.value}/{total_length} done.")
        finally:
            lock.release()


def multi_process(jsonpaths, args):
    process_num = args.num_worker
    # check if is valid

    # start process
    print("To deal {} videos in {} process".format(len(jsonpaths), args.num_worker))
    files = jsonpaths
    grid_size = len(files) // process_num
    process_pool = []
    lock = Lock()
    counter = Value("i", 0)
    for i in range(process_num):
        start_index = grid_size * i
        if i != process_num - 1:
            end_index = grid_size * (i + 1)
        else:
            end_index = len(files)
        pw = Process(target=process_paths,
                     args=(files[start_index:end_index], args, lock, counter, len(files)))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

def main():
    args = parse_args()
    if not os.path.exists(args.out_json_dir):
        os.makedirs(args.out_json_dir)

    # get all video folders
    jsonpaths = glob.glob(os.path.join(args.src_folder, str(Path('*/'*args.level))+'*.json'))

    # check if dir is dealed
    print("Found {} videos! {} videos not yet processed!".format(len(jsonpaths), len(jsonpaths)))

    # multi process
    multi_process(jsonpaths, args)

import multiprocessing

multiprocessing.set_start_method('forkserver', force=True)
if __name__ == '__main__':
    main()