import json
import os
import sys
import numpy as np
import struct
import shutil

fracbits = 15
json_path = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images/train_09.json'
out_bin_dir = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images/train_09bins'

if os.path.exists(out_bin_dir):
    shutil.rmtree(out_bin_dir)

if not os.path.exists(out_bin_dir):
    os.makedirs(out_bin_dir)

def write_features_to_bin(f, out_bin_path):
    # [32, 128]
    f = np.array(f)

    # [32, 128] --> [128, 32] --> [128*32]
    f = f.swapaxes(0, 1).flatten()

    # save to bin
    
    int16_flist = [int(v * (2**fracbits)) for v in f]
    # int16_flist = [max(v, (-0x7fff-1)) for v in int16_flist]
    # int16_flist = [min(v, 0x7fff) for v in int16_flist]
    with open(out_bin_path, 'wb') as wp:
        try:
            wp.write(struct.pack('<{}h'.format(len(int16_flist)), *int16_flist))
        except:
            print(out_bin_path)
        #     for num in int16_flist:
        #         if num > 0x7fff or num < (-0x7fff-1):
        #             print(num)
        #             print(-0x7fff)

with open(json_path, 'r') as fp:
    features = json.load(fp)
    print(len(features))
    for vprefix in features:
        f = features[vprefix]
        out_bin_path = os.path.join(out_bin_dir, '_'.join(vprefix.split('/'))+'.bin')
        write_features_to_bin(f, out_bin_path)




