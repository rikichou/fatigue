import sys
import os
print(sys.platform)
if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    sys.path.append("/Users/zhourui/workspace/pro/source/yolov5")
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    sys.path.append(r"D:\workspace\pro\source\yolov5")
else:
    print('=>>>>load data from linux platform')
    sys.path.append(r"/zhourui/workspace/pro/source/yolov5")

import glob
import argparse
from multiprocessing import Process, Lock, Value
import cv2
from cv2 import VideoWriter_fourcc
import numpy as np
from pathlib import Path
import mmcv

from FaceDetection import FaceDetect

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='video dir')
    parser.add_argument(
        'out_dir', type=str, help='out video dir')
    parser.add_argument(
        '--size', type=int, default=112, help='out video dir')
    parser.add_argument(
        '--level',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='directory level of data')
    parser.add_argument(
        '--ext',
        type=str,
        default='avi',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--num_worker',
        type=int,
        default=10,
        help='num workers to preprocess')
    parser.add_argument('--weights',
                        default=r'D:\workspace\pro\source\yolov5\weights\200_last.pt',
                        help='experiment', type=str)
    parser.add_argument('--imgsz',
                        default=640,
                        help='experiment', type=int)
    parser.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help='whether to use cpu')

    args = parser.parse_args()
    return args

def process_videos(videos, args, lock, counter, total_length):
    # init face detection model
    faceDetect = FaceDetect(args=args)

    def get_input_face(image, rect):
        sx, sy, ex, ey = rect
        h, w, c = image.shape
        faceh = ey - sy
        facew = ex - sx

        longsize = max(faceh, facew)
        expendw = longsize - facew
        expendh = longsize - faceh

        sx = sx - (expendw / 2)
        ex = ex + (expendw / 2)
        sy = sy - (expendh / 2)
        ey = ey + (expendh / 2)

        sx = int(max(0, sx))
        sy = int(max(0, sy))
        ex = int(min(w - 1, ex))
        ey = int(min(h - 1, ey))

        return image[sy:ey, sx:ex, :]

    for v in videos:
        # open video
        vr = mmcv.VideoReader(v)
        fps = vr.fps
        # video writer
        out_video_file = os.path.join(args.out_dir, 'face_'+os.path.basename(v))
        fourcc = 'XVID'
        vwriter = cv2.VideoWriter(out_video_file, VideoWriter_fourcc(*fourcc), fps)

        for i, vr_frame in enumerate(vr):
            if vr_frame is not None:
                # face detect
                image = vr_frame
                bbox = faceDetect.detect(image)
                # print(bbox)
                if len(bbox) != 4 or sum(bbox) < 1:
                    continue
                else:
                    sx, sy, ex, ey = bbox
                    image = get_input_face(image, [sx,sy,ex,ey])
                    image = cv2.resize(image, (args.size, args.size))
                    vwriter.write(image)
            else:
                print(
                    'Length inconsistent!'
                    f'Early stop with {i + 1} out of {len(vr)} frames.')
                break
        # release writer
        vwriter.release()

        # counter
        lock.acquire()
        try:
            # p_bar.update(1)
            counter.value += 1
            if counter.value % 50 == 0:
                print(f"{counter.value}/{total_length} done.")
        finally:
            lock.release()


def multi_process(video_dirs, args):
    process_num = args.num_worker
    # check if is valid
    to_process_videos = []
    for v in video_dirs:
        if not os.path.isfile(v):
            continue
        to_process_videos.append(v)

    # start process
    files = to_process_videos
    grid_size = len(files) // process_num
    process_pool = []
    lock = Lock()
    counter = Value("i", 0)
    for i in range(process_num):
        start_index = grid_size * i
        if i != process_num - 1:
            end_index = grid_size * (i + 1)
        else:
            end_index = len(files)
        pw = Process(target=process_videos,
                     args=(files[start_index:end_index], args, lock, counter, len(files)))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

def main():
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # get all video folders
    videos = glob.glob(os.path.join(args.src_folder, str(Path('*/'*args.level))+'.'+args.ext))
    print("Found {} videos!".format(len(videos)))

    # multi process
    multi_process(videos, args)

if __name__ == '__main__':
    main()