import cv2
import os
import json

IMG_ROOT_DIR = r'E:\360Downloads\ti\BY2399_20210207_211228_C02_Main_02_02'
JSON_FILE = r'E:\360Downloads\ti\BY2399_20210207_211228_C02_Main_02_02.json'

with open(JSON_FILE, 'r') as fp:
    anns = json.load(fp)

    for img in anns:
        imgpath = os.path.join(IMG_ROOT_DIR, img)
        image = cv2.imread(imgpath)

        rect = anns[img]['facerect']
        points = anns[img]['mmpose_kpts']

        if rect is None:
            continue
        sx,sy,ex,ey = rect
        cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 10)
        for idx,p in enumerate(points):
            x, y, score = p
            x = x+sx
            y = y+sy
            cv2.circle(image, (int(x), int(y)), 1, (0,255,255), 2)

        cv2.imshow('1', image)
        cv2.waitKey(10)