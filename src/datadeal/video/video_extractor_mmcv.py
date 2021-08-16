import os
import cv2

import mmcv

VIDEO_PATH = '/Users/zhourui/workspace/pro/fatigue/data/C5452600000000-210531-083518-083528-01p016000000.avi'
OUT_RAWFRAMES_DIR = '/Users/zhourui/workspace/pro/fatigue/data/out/test_mmcv_gray_C5452600000000-210531-083518-083528-01p016000000.avi'

if not os.path.exists(OUT_RAWFRAMES_DIR):
    os.makedirs(OUT_RAWFRAMES_DIR)

vr = mmcv.VideoReader(VIDEO_PATH)

for idx, frame in enumerate(vr):
    mmcv.imwrite(frame, f'{OUT_RAWFRAMES_DIR}/img_{idx + 1:05d}.jpg')