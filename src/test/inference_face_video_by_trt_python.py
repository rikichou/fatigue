import os
import time
from multiprocessing import Process, Lock, Value

from pathlib import Path
import argparse
import glob

from fatigue_face_video_tensorrt_python import fatigue

test_video_path = '/home/ruiming/workspace/pro/fatigue/data/test/video/lookdown_face/face_02_03_0_0_28716582.avi'

#model_path = '/home/ruiming/workspace/pro/source/mmaction2/work_dirs/fatigue_r50_clean/fatigue_r50_clean_fp16.trt'
model_path = '/home/ruiming/workspace/pro/source/mmaction2/work_dirs/fatigue_r50_clean_withnormal/fatigue_r50_clean_withnormal_fp16.trt'
config_path = '/home/ruiming/workspace/pro/source/mmaction2/configs/recognition/csn/fatigue_r50_clean_inference.py'

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_folder',
                        type=str,
                        help='face video path')
    parser.add_argument('--model_path', default='fatigue_face_video_tensorrt_python/model/fatigue_r50_clean_withnormal/fatigue_r50_clean_withnormal_fp16.trt', type=str, help='tensorrt model path')
    parser.add_argument('--config_path', default='fatigue_face_video_tensorrt_python/model/fatigue_r50_clean_withnormal/fatigue_r50_clean_inference.py', type=str, help='mmaction inference config path')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2, 3],
        default=3,
        help='directory level of data')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=1,
        help='number of workers to inference')
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

    args = parser.parse_args()

    return args

def process_videos(videos, args, results, lock, counter, total_length):

    # model init
    fh = fatigue.FatigueFaceVideoTensorrt(args.model_path, args.config_path)

    for v in videos:
        # inference
        pred = fh(v)

        # counter
        lock.acquire()
        try:
            # update results
            results[v] = pred
            # p_bar.update(1)
            counter.value += 1
            if counter.value % 50 == 0:
                print(f"{counter.value}/{total_length} done.")
        finally:
            lock.release()

def multi_process(videos, args):
    process_num = args.num_worker
    # check if is valid
    to_process_videos = []
    for v in videos:
        if not os.path.isfile(v):
            continue
        to_process_videos.append(v)

    # start process
    results = {}
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
                     args=(files[start_index:end_index], args, results, lock, counter, len(files)))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

    print(results)

def main():
    args = parse_args()

    # get all video folders
    videos = glob.glob(os.path.join(args.src_folder, str(Path('*/'*args.level))+'.'+args.ext))
    videos = videos[:10]
    print("Found {} videos!".format(len(videos)))

    # multi process
    st = time.time()
    multi_process(videos, args)
    tcost = time.time()-st
    print("Dealed {} videos cost {:.4f}(average {:.4f}s/video)".format(len(videos), tcost, tcost/len(videos)))
if __name__ == '__main__':
    main()
