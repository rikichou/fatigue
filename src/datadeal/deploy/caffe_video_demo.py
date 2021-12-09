import sys
import os
import cv2
import numpy as np
from fatigue_caffe_python import fatigue_cnn_cnn as fatigue

# face detection handle
yolov5_src = "/home/ruiming/workspace/pro/source/yolov5"
sys.path.append(yolov5_src)
from FaceDetection import FaceDetect
class Obj(object):
    def __init__(self):
        super().__init__()
args = Obj()
args.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
args.imgsz = 640
args.cpu = False
fd = FaceDetect(args=args)

# create fatigue handle
backbone_def = 'fatigue_caffe_python/model/slowfast_r50_3d_4x16x1_256e_clips_rgb_2021-11-25-15-08-01_tiny/backbone_debug_112_sim.prototxt'
backbone_weights = 'fatigue_caffe_python/model/slowfast_r50_3d_4x16x1_256e_clips_rgb_2021-11-25-15-08-01_tiny/backbone_debug_112_sim.caffemodel'

clshead_def = 'fatigue_caffe_python/model/slowfast_r50_3d_4x16x1_256e_clips_rgb_2021-11-25-15-08-01_tiny/clshead_debug_sim.prototxt'
clshead_weights = 'fatigue_caffe_python/model/slowfast_r50_3d_4x16x1_256e_clips_rgb_2021-11-25-15-08-01_tiny/clshead_debug_sim.caffemodel'

fatd = fatigue.FatigueCnnCnn(backbone_def=backbone_def, backbone_weights=backbone_weights,
                            clshead_def=clshead_def, clshead_weights=clshead_weights)

def video_demo(video_path, fd, fatd):
    cap = cv2.VideoCapture(video_path)

    input_features = []
    seq_len = 32
    jit_size = 2
    frame_idx = 0

    last_f = None
    while True:
        frame_idx += 1
        # read image from video
        ret, frame = cap.read()
        if ret is False:
            break
        if frame_idx%jit_size != 0:
            continue
        # get face rectangle
        image = frame.copy()
        bbox = fd.detect(image)
        if len(bbox) != 4 or sum(bbox) < 1:
            print("cat not detect face")
            continue
        else:
            sx, sy, ex, ey = bbox
        # get face features
        face_rect = (int(sx), int(sy), int(ex), int(ey))
        feature = fatd.get_feature(image, face_rect)

        vector1 = last_f
        vector2 = feature
        if not last_f is None:
            op7=np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
            #print(op7)
        last_f = feature
        
        # check if features len is equal to seq_len
        if len(input_features)<=seq_len:
            input_features.append(feature.copy())
            print(input_features)
        # predict
        prob = 0
        if len(input_features)>=seq_len:
            # remove first feature
            if len(input_features)>seq_len:
                input_features = input_features[1:]
            prob = fatd(np.array(input_features))
        print(prob)
        # debug
        sx,sy,ex,ey = fatd.inputface_rect
        cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 10)
        cv2.putText(image, '{:.3f}'.format(prob), (sx, sy - 5),
                    0, 1, (0, 0, 255), 1)
        cv2.imshow('debug', image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

video_path = '/home/ruiming/workspace/pro/fatigue/data/test/video/1103/97f3cfa5fc0b293673aeed04f839e387.mp4'
video_path = '/home/ruiming/workspace/pro/fatigue/src/datadeal/deploy/fatigue_caffe_python/data/videos/A05500D0000000-210528-134735-134740-01p016000000.avi'
video_path = '/home/ruiming/workspace/pro/fatigue/src/datadeal/deploy/fatigue_caffe_python/data/videos/A5632000000000-210528-140025-140035-01p016000000.avi'
#video_path = '/home/ruiming/workspace/pro/fatigue/src/datadeal/deploy/fatigue_caffe_python/data/videos/002B0000FD000000-210419-104008-104012-01p212000000.avi'
#video_path = '/home/ruiming/workspace/pro/fatigue/src/datadeal/deploy/fatigue_caffe_python/data/videos/晋A02317D-134440-211101-194018-194027-11p21B000000.avi'
video_demo(video_path, fd, fatd)
