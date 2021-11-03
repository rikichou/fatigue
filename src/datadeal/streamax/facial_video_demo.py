import sys
import os
sys.path.append('/Users/zhourui/workspace/pro/smoke_keypoint/src/common_utils')
yolov5_src = "/Users/zhourui/workspace/pro/source/yolov5"
sys.path.insert(0, yolov5_src)
import numpy as np
from mmpose.apis import (inference_bottom_up_pose_model,
                         inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
import cv2

video_path = '/Users/zhourui/Downloads/test/02_02_6501_0_7dbdb083eba04517a80d0b4479b0f87f.avi'

def test_top_down_demo():
    # Face300WDataset demo
    person_result = []
    person_result.append({'bbox': [50, 50, 50, 100]})
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        '/Users/zhourui/workspace/pro/source/mmpose/configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/300w/hrnetv2_w18_300w_256x256.py',
        '/Users/zhourui/workspace/pro/source/mmpose/pretrained/face/hrnetv2_w18_300w_256x256-eea53406_20211019.pth',
        device='cpu')
    dataset_info = DatasetInfo(pose_model.cfg.data['test'].get(
        'dataset_info', None))

    # init face detection
    class TmpClass(object):
        def __init__(self):
            super().__init__()

    args = TmpClass()

    from FaceDetection import FaceDetect
    args.cpu = True
    args.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
    args.imgsz = 640
    fd = FaceDetect(args=args)

    if True:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
    while True:
        # read image
        ret, frame = cap.read()

        if frame is None:
            continue

        # face detection
        image = frame

        bbox = fd.detect(image)
        if len(bbox) != 4 or sum(bbox) < 1:
            print("cat not detect face")
            continue
        else:
            sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]

        # test a single image, with a list of bboxes.
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            image,
            None,
            format='xywh',
            dataset_info=dataset_info)
        # show the results
        img = vis_pose_result(
            pose_model, image, pose_results, dataset_info=dataset_info)

        cv2.imshow('1', img)
        cv2.waitKey(1)

test_top_down_demo()