import json
import os
import glob
import numpy as np

npy_dir = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian_outkeypoints_multi'
out_json_dir = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian_json'
if not os.path.exists(out_json_dir):
    os.makedirs(out_json_dir)

labels = glob.glob(npy_dir+r'\*.npy')

for idx,l in enumerate(labels):
    # read label info
    info = np.load(l, allow_pickle=True).item()

    out_json_path = os.path.join(out_json_dir, os.path.splitext(os.path.basename(l))[0] + '.json')

    out_json_info = {}
    for imginfo in info:
        out_json_info[imginfo] = {}
        out_json_info[imginfo]['bbox'] = info[imginfo]['facerect']

    with open(out_json_path, 'w') as f:
        json.dump(out_json_info, f)

    if idx%1000 == 0:
        print("{}/{}".format(idx, len(labels)))