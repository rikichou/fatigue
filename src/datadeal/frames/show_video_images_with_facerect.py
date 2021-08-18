import os
import glob
import numpy as np
import cv2

IMG_DIR = '/Users/zhourui/workspace/pro/fatigue/pict/pictest/close/v1'

label_file_path = os.path.join(IMG_DIR, 'facerect.npy')
infos_dict = np.load(label_file_path, allow_pickle=True).item()

for item in infos_dict:
    sx,sy,ex,ey = infos_dict[item]
    img_path = os.path.join(IMG_DIR, item)
    image = cv2.imread(img_path)

    cv2.rectangle(image, (sx,sy), (ex,ey), (255,0,0), 1)
    cv2.imshow('1', image)
    cv2.waitKey(0)


