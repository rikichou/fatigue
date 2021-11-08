import os
import numpy as np
import glob
import cv2
import json

label_file = r'H:\pro\fatigue\data\dahaqian\20211108_fatigue_dahaqian.json'
video_dir = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian'

with open(label_file, 'r') as f:
    infos = json.load(f)

def check_if_index(frame_idx, index_list):
    for idx in index_list:
        if frame_idx in range(idx-10, idx):
            return True
    return False

for vprefix in infos:
    info = infos[vprefix]
    # get video
    vname = vprefix.split('/')[1] + '.avi'
    vpath = os.path.join(video_dir, vname)

    # dahaqian index
    dahaqian_indexes = info['fatigue_warning_idx']                    

    cap = cv2.VideoCapture(vpath)

    frame_idx = 0
    while True:
        frame_idx += 1
        # read image
        ret, frame = cap.read()
        if frame is None:
            break
        # show
        color = (0,255,0)
        if check_if_index(frame_idx, dahaqian_indexes):
            color = (0,0,255)
        cv2.rectangle(frame, (100, 100), (300, 300), color, 1)

        cv2.imshow('1', frame)
        if cv2.waitKey(0) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()