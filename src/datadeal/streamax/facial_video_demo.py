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

def test_top_down_demo():
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

    video_path = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian\02_02_6501_0_3bdbea23d7294e068cb0480945ec324e.avi'
    video_dir = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian'

    videos = os.listdir(video_dir)
    for v in videos:
        if '.avi' not in v:
            continue
        video_path = os.path.join(video_dir, v)
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
            # img = vis_pose_result(
            #     pose_model, image, pose_results, dataset_info=dataset_info)
            points =pose_results[0]['keypoints']

            l = points[60]
            r = points[64]
            t = points[62]
            b = points[66]

            w = r[0] - l[0]
            h = b[1] - t[1]

            ratio = h / w
            color = (0, 255, 0)
            if ratio > 0.5:
                color = (0,0,255)

            for idx,kpt in enumerate(points):
                if idx <=67 and idx >= 60:
                    x,y,score = kpt
                    cv2.circle(image, (int(x),int(y)), 1, color, 2)

            #print(pose_results)
            cv2.imshow('1', image)
            if cv2.waitKey(1) == ord('q'):
                break

test_top_down_demo()