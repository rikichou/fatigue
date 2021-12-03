import json
import os
import sys
import numpy as np
import struct

fracbits = 13
json_path = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images/train.json'
out_bin_dir = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images/trainbins'

if not os.path.exists(out_bin_dir):
    os.makedirs(out_bin_dir)

def write_features_to_bin(f, out_bin_path):
    # [32, 128]
    f = np.array(f)

    # [32, 128] --> [128, 32] --> [128*32]
    f = f.swapaxes(0, 1).flatten()

    # save to bin
    int16_flist = [int(v * (2**fracbits)) for v in f]
    with open(out_bin_path, 'wb') as wp:
        wp.write(struct.pack('<{}h'.format(len(int16_flist)), *int16_flist))

with open(json_path, 'r') as fp:
    features = json.load(fp)

    for vprefix in features:
        f = features[vprefix]
        out_bin_path = os.path.join(out_bin_dir, '_'.join(vprefix.split('/'))+'.bin')
        write_features_to_bin(f, out_bin_path)




