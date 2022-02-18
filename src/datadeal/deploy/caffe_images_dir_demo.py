import sys
import os
import os

import cv2
import numpy as np
from fatigue_caffe_python import fatigue_cnn_cnn as fatigue

# create fatigue handle
backbone_def = '/home/ruiming/workspace/pro/source/ONNXToCaffe/out/fatigue/board/twostage_tinyresnet18_32_128e112_bgr_fatigue_embedding/20211227/backbone_112_embedding_sim.prototxt'
backbone_weights = '/home/ruiming/workspace/pro/source/ONNXToCaffe/out/fatigue/board/twostage_tinyresnet18_32_128e112_bgr_fatigue_embedding/20211227/backbone_112_embedding_sim.caffemodel'

clshead_def = '/home/ruiming/workspace/pro/source/ONNXToCaffe/out/fatigue/board/twostage_tinyresnet18_32_128e112_bgr_fatigue_embedding/20211227/clshead_112_embedding_sim.prototxt'
clshead_weights = '/home/ruiming/workspace/pro/source/ONNXToCaffe/out/fatigue/board/twostage_tinyresnet18_32_128e112_bgr_fatigue_embedding/20211227/clshead_112_embedding_sim.caffemodel'

fatd = fatigue.FatigueCnnCnn(backbone_def=backbone_def, backbone_weights=backbone_weights,
                            clshead_def=clshead_def, clshead_weights=clshead_weights)

format_idx = True
def video_demo(video_dir, fatd):

    input_features = []
    seq_len = 32
    frame_idx = 0

    # read all images
    imgs = os.listdir(video_dir)
    if len(imgs) != seq_len:
        print("Invalid dir {}, has {} imgs".format(d, len(imgs)))
        return
    if format_idx:
        img_idxs = [int(n[4:9]) for n in imgs]
    else:
        img_idxs = list(range(len(imgs)))
    img_idxs.sort()

    # read images and get features
    input_features = []
    for idx in img_idxs:
        if format_idx:
            img_path = os.path.join(video_dir, 'img_{:05d}.jpg'.format(idx))
        else:
            img_path = os.path.join(video_dir, '{}.jpg'.format(idx))
        # read image from video
        frame = cv2.imread(img_path)
        if frame is False:
            print("Failed to read ", img_path)
            break
        image = frame.copy()
        # get face rectangle
        sx,sy,ex,ey = 0,0,image.shape[1],image.shape[0]
        # get face features
        face_rect = (int(sx), int(sy), int(ex), int(ey))
        
        feature = fatd.get_feature(image, face_rect)
        #print(feature)
        
        input_features.append(feature.copy())

    # check features
    if len(input_features) != 32:
        print("No enough featuers ", len(input_features))
        return 

    prob = fatd(np.array(input_features))
    print(input_features[0])
    print(img_idxs[0])
    print(prob)
        # debug
#     sx,sy,ex,ey = fatd.inputface_rect
#     cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 10)
#     cv2.putText(image, '{:.3f}'.format(prob), (sx, sy - 5),
#                 0, 1, (0, 0, 255), 1)
#     cv2.imshow('debug', image)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break
# cv2.destroyAllWindows()


video_dir = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images_20211227/test/UN/yueBDE77000000000-210817-000047-000057-01p012000000'
#video_dir = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images/test/CE/02_65_6501_0_7bf59ef7aea84652bd1df320230b21c6'
video_demo(video_dir, fatd)
