import os

import cv2
from PIL import Image
import numpy as np

VIDEO_PATH = '/Users/zhourui/workspace/pro/fatigue/data/C5452600000000-210531-083518-083528-01p016000000.avi'
OUT_RAWFRAMES_DIR = '/Users/zhourui/workspace/pro/fatigue/data/out/test_pillow_gray_C5452600000000-210531-083518-083528-01p016000000.avi'
if not os.path.exists(OUT_RAWFRAMES_DIR):
    os.makedirs(OUT_RAWFRAMES_DIR)

cap = cv2.VideoCapture(VIDEO_PATH)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR color
    #cv2.imwrite(f'{OUT_RAWFRAMES_DIR}/img_{idx + 1:05d}.jpg', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    im = Image.fromarray(np.uint8(frame))
    im.save(f'{OUT_RAWFRAMES_DIR}/img_{idx + 1:05d}.jpg')

    idx += 1

cap.release()




