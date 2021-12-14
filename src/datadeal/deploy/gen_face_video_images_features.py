import sys
import os
import json
import glob
import numpy as np

seq_len = 32
video_root_dir = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images/test'
out_json_path = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images/test_01111.json'
if os.path.exists(out_json_path):
    os.remove(out_json_path)

import cv2
from fatigue_caffe_python import fatigue_cnn_cnn as fatigue

# create fatigue handle
backbone_def = '/home/ruiming/workspace/pro/source/ONNXToCaffe/out/fatigue/board/backbone_embedding_sim.prototxt'
backbone_weights = '/home/ruiming/workspace/pro/source/ONNXToCaffe/out/fatigue/board/backbone_embedding_sim.caffemodel'

clshead_def = '/home/ruiming/workspace/pro/source/ONNXToCaffe/out/fatigue/board/clshead_embedding_sim.prototxt'
clshead_weights = '/home/ruiming/workspace/pro/source/ONNXToCaffe/out/fatigue/board/clshead_embedding_sim.caffemodel'

fatd = fatigue.FatigueCnnCnn(backbone_def=backbone_def, backbone_weights=backbone_weights,
                            clshead_def=clshead_def, clshead_weights=clshead_weights)

files = glob.glob(os.path.join(video_root_dir, '*/*'))
dirs = [f for f in files if os.path.isdir(f)]

out_json_info = {}
for count,d in enumerate(dirs):
    # get indexs and sort
    imgs = os.listdir(d)
    if len(imgs) != seq_len:
        print("Invalid dir {}, has {} imgs".format(d, len(imgs)))
        continue
    img_idxs = [int(n[4:9]) for n in imgs]
    img_idxs.sort()

    # read images and get features
    features = []
    for idx in img_idxs:
        img_path = os.path.join(d, 'img_{:05d}.jpg'.format(idx))

        # image read and preprocess
        image = cv2.imread(img_path)

        # get face feature

        feature = fatd.get_feature(image, [0,0,image.shape[1],image.shape[0]])
        if '02_03_0_0_28076958' in d:
            print("idx {}, rect {}".format(idx, [0,0,image.shape[1],image.shape[0]]))
            print(feature)
        feature = np.array(feature).reshape(-1).tolist()

        features.append(feature.copy())
    
    vinfo = '/'.join(d.rsplit('/', maxsplit=2)[1:])
    out_json_info[vinfo] = features

    if count % 10 == 0:
        print("{}/{}".format(count, len(dirs)))

# save json
with open(out_json_path, 'w') as f:
    json.dump(out_json_info, f)