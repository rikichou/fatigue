import os
import numpy as np
import glob
import argparse
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser(description='get facerect')
    parser.add_argument(
        'v_dir', type=str, help='root directory for the frames')
    parser.add_argument(
        'out_dir', type=str, help='root directory for the frames')
    parser.add_argument(
        '--level',
        type=int,
        default=1,
        help='directory level of data')

    args = parser.parse_args()
    return args

def get_facerects(filepath):
    facerect_infos = {}
    with open(filepath, 'r') as fp:
        anns = json.load(fp)
        for imgname in anns:
            facerect_infos[imgname] = anns[imgname]['bbox']
    return facerect_infos

def main():
    args = parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # get all video folders
    video_facerects = glob.glob(os.path.join(args.src_folder, str(Path('*/'*args.level)), "facerect.npy"))

    # get json file
    for idx,vf in enumerate(video_facerects):
        json_info = {}
        # get face rect info
        facerect_infos = np.load(vf, allow_pickle=True).item()
        for imgname in facerect_infos:
            json_info[imgname] = {}
            json_info[imgname]['bbox'] = facerect_infos[imgname]
        # write json info
        out_json_filepath = os.path.join(args.out_dir, vf.rsplit('/', maxsplit=2)[-2]+'.json')

        with open(out_json_filepath, 'w') as f:
            json.dump(json_info, f)


        if idx%1000 == 0:
            print("Dealed {}/{}".format(idx,len(video_facerects)))

if __name__ == '__main__':
    main()