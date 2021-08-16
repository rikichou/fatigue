import os

import cv2

VIDEO_PATH = '/Users/zhourui/workspace/pro/fatigue/data/A8079900000000-210528-135309-135314-01p016000000.avi'
OUT_RAWFRAMES_DIR = '/Users/zhourui/workspace/pro/fatigue/data/out/test_opencv_gray'
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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{OUT_RAWFRAMES_DIR}/img_{idx + 1:05d}.jpg', frame)

    idx += 1

cap.release()

# cap = cv2.VideoCapture(VIDEO_PATH)
#
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     print(frame, ret)
#     cv2.imshow('image', frame)
#     k = cv2.waitKey(20)
#     # q键退出
#     if (k & 0xff == ord('q')):
#         break
#
# cap.release()
# cv2.destroyAllWindows()