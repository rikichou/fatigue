import sys
import os

yolov5_src = r"D:\workspace\pro\source\yolov5"

sys.path.insert(0, yolov5_src)
import numpy as np
from mmpose.apis import (inference_bottom_up_pose_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
import cv2
import mmcv
from multiprocessing import Process, Lock, Value


def test_top_down_demo(videos, lock, counter, total_length):
    # Face300WDataset demo
    person_result = []
    person_result.append({'bbox': [50, 50, 50, 100]})
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        r'E:\workspace\pro\source\mmpose\configs\face\2d_kpt_sview_rgb_img\topdown_heatmap\300w\hrnetv2_w18_300w_256x256.py',
        r'E:\workspace\pro\source\mmpose\pretrained\face\hrnetv2_w18_300w_256x256-eea53406_20211019.pth',
        device='cuda')
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))

    # init face detection
    class TmpClass(object):
        def __init__(self):
            super().__init__()

    args = TmpClass()

    from FaceDetection import FaceDetect
    args.cpu = False
    args.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
    args.imgsz = 640
    fd = FaceDetect(args=args)

    for idx,v in enumerate(videos):
        if '.avi' not in v:
            continue
        video_path = v
        vr = mmcv.VideoReader(video_path)
        # for i in range(len(vr)):
        points_info = {}
        for i, vr_frame in enumerate(vr):
            if vr_frame is not None:
                # face detection
                image = vr_frame
                imgname = 'img_{:05d}.jpg'.format(i + 1)

                points_info[imgname] = {}
                bbox = fd.detect(image)
                if len(bbox) != 4 or sum(bbox) < 1:
                    print("cat not detect face")
                    points_info[imgname]['facerect'] = None
                    continue
                else:
                    sx, sy, ex, ey = bbox
                    points_info[imgname]['facerect'] = [sx,sy,ex,ey]
                image = image[sy:ey, sx:ex]

                # test a single image, with a list of bboxes.
                pose_results, _ = inference_top_down_pose_model(
                    pose_model,
                    image,
                    None,
                    format='xywh',
                    dataset_info=dataset_info)
                # show the results
                # img = vis_pose_result(
                #     pose_model, image, pose_results, dataset_info=dataset_info)
                points = pose_results[0]['keypoints']
                points_info[imgname]['points'] = points
            else:
                print(
                    'Length inconsistent!'
                    f'Early stop with {i + 1} out of {len(vr)} frames.')
                break
        out_dir = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian_outkeypoints_multi'
        np.save(os.path.join(out_dir, os.path.splitext(os.path.basename(v))[0] + '.npy'), points_info)
        # counter
        lock.acquire()
        try:
            # p_bar.update(1)
            counter.value += 1
            if counter.value % 50 == 0:
                print(f"{counter.value}/{total_length} done.")
        finally:
            lock.release()

def multi_process(video_dirs):
    process_num = 6

    # start process
    files = video_dirs
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
        pw = Process(target=test_top_down_demo,
                     args=(files[start_index:end_index], lock, counter, len(files)))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

import glob
from pathlib import Path
import multiprocessing

# multiprocessing.set_start_method('forkserver', force=True)

def main():
    video_dir = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian'
    out_dir = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian_outkeypoints_multi'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    videos = glob.glob(os.path.join(video_dir, str(Path('*/' * 1)) + '.avi'))

    multi_process(videos)

if __name__ == '__main__':
    main()
