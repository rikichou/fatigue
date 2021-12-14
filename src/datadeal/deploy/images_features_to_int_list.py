import json
import os
import sys
import numpy as np
import struct
import shutil

fracbits = 12
json_path = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images/test_09.json'
out_fpath = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images/test_09_features_int.txt'

def features_convert(f):
    # [32, 128]
    f = np.array(f)

    # [32, 128] --> [128, 32] --> [128*32]
    f = f.swapaxes(0, 1).flatten()
    f = [int(x*4096) for x in f]

    f = [str(num)+',' for num in f]

    f[-1] = f[-1][:-1]

    f.append('\n')

    return f

with open(json_path, 'r') as fp:
    features = json.load(fp)
    print(len(features))

    with open(out_fpath, 'w') as fp:
        for vprefix in features:
            ffloat = features[vprefix]
            f = features_convert(ffloat)

            fp.writelines(f)






