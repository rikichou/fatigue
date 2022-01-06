import os
import numpy as np
import glob
import cv2

label_dir = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian_outkeypoints_multi'
video_dir = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian'

labels = glob.glob(label_dir+r'\*.npy')
    
for l in labels:
    # read label info
    info = np.load(l, allow_pickle=True).item()

    # get video
    vname = os.path.splitext(os.path.basename(l))[0]+'.avi'
    vpath = os.path.join(video_dir, vname)

    cap = cv2.VideoCapture(vpath)

    frame_idx = 0
    while True:
        frame_idx += 1
        # read image
        ret, frame = cap.read()

        if frame is None:
            break

        # get image label
        imgname = 'img_{:05d}.jpg'.format(frame_idx)
        if imgname not in info:
            print("{} not in info! {}".format(imgname, info.keys()))
            continue

        imginfo = info[imgname]
        if imginfo['facerect'] == None:
            print("Face rect is None!")
            cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 255), 1)
        else:
            # face rectangle
            sx, sy, ex, ey = imginfo['facerect']
            sx = int(sx)
            sy = int(sy)
            ex = int(ex)
            ey = int(ey)
            cv2.rectangle(frame, (sx,sy), (ex,ey), (255,0,0), 1)

            # points
            points = imginfo['points']
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
                    cv2.circle(frame, (int(x),int(y)), 1, color, 2)

        cv2.imshow('1', frame)
        if cv2.waitKey(0) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()