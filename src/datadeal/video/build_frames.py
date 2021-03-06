import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Pool

import tqdm
import cv2
from PIL import Image
import mmcv

import numpy as np


def extract_frame(vid_item):
    """Generate optical flow using dense flow.

    Args:
        vid_item (list): Video item containing video full path,
            video (short) path, video id.

    Returns:
        bool: Whether generate optical flow successfully.
    """
    full_path, vid_path, vid_id = vid_item
    if '/' in vid_path:
        act_name = osp.basename(osp.dirname(vid_path))
        out_full_path = osp.join(args.out_dir, act_name)
    else:
        out_full_path = args.out_dir

    # Not like using denseflow,
    # Use OpenCV will not make a sub directory with the video name
    video_name = osp.splitext(osp.basename(vid_path))[0]
    out_full_path = osp.join(out_full_path, video_name)
    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)

    vr = mmcv.VideoReader(full_path)
    # for i in range(len(vr)):
    for i, vr_frame in enumerate(vr):
        if vr_frame is not None:
            # w, h, _ = np.shape(vr_frame)
            # if args.new_short == 0:
            #     if args.new_width == 0 or args.new_height == 0:
            #         # Keep original shape
            #         out_img = vr_frame
            #     else:
            #         out_img = mmcv.imresize(vr_frame,
            #                                 (args.new_width,
            #                                  args.new_height))
            # else:
            #     if min(h, w) == h:
            #         new_h = args.new_short
            #         new_w = int((new_h / h) * w)
            #     else:
            #         new_w = args.new_short
            #         new_h = int((new_w / w) * h)
            #     out_img = mmcv.imresize(vr_frame, (new_h, new_w))
            # mmcv.imwrite(vr_frame,
            #              f'{out_full_path}/img_{i + 1:05d}.jpg')
            frame = cv2.cvtColor(vr_frame, cv2.COLOR_BGR2GRAY)
            im = Image.fromarray(np.uint8(frame))
            im.save(f'{out_full_path}/img_{i + 1:05d}.jpg')
        else:
            warnings.warn(
                'Length inconsistent!'
                f'Early stop with {i + 1} out of {len(vr)} frames.')
            break

    sys.stdout.flush()
    return True

def is_video(name):
    _,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in ['.avi', '.mp4', '.webm']:
        return True
    else:
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2],
        default=2,
        help='directory level of data')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='number of workers to build rawframes')
    parser.add_argument(
        '--ext',
        type=str,
        default='avi',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--mixed-ext',
        action='store_true',
        help='process video files with mixed extensions')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new-short',
        type=int,
        default=0,
        help='resize image short side length keeping ratio')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    if args.level == 2:
        classes = os.listdir(args.src_dir)
        for classname in classes:
            new_dir = osp.join(args.out_dir, classname)
            if not osp.isdir(new_dir):
                print(f'Creating folder: {new_dir}')
                os.makedirs(new_dir)


    print('Reading videos from folder: ', args.src_dir)
    if args.mixed_ext:
        print('Extension of videos is mixed')
        fullpath_list = glob.glob(args.src_dir + '/*' * args.level)
        done_fullpath_list = glob.glob(args.out_dir + '/*' * args.level)
        fullpath_list = [x for x in fullpath_list if is_video(x)]
    else:
        print('Extension of videos: ', args.ext)
        fullpath_list = glob.glob(args.src_dir + '/*' * args.level + '.' +
                                  args.ext)
        done_fullpath_list = glob.glob(args.out_dir + '/*' * args.level)
    print('Total number of videos found: ', len(fullpath_list))

    if args.level == 2:
        vid_list = list(
            map(
                lambda p: osp.join(
                    osp.basename(osp.dirname(p)), osp.basename(p)),
                fullpath_list))
    elif args.level == 1:
        vid_list = list(map(osp.basename, fullpath_list))

    with Pool(args.num_worker) as pool:
        r = list(tqdm.tqdm(pool.imap(
            extract_frame,
            zip(fullpath_list, vid_list, range(len(vid_list))))))
