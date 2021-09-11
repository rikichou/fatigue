import os
import argparse
import shutil
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Lock, Value

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('label_file', type=str, help='label_file')
    parser.add_argument('results_file', type=str, help='results_file')
    #parser.add_argument('data_root_dir', type=str, help='data_root_dir')
    parser.add_argument('out_root_dir', type=str, help='out_root_dir')
    parser.add_argument(
        '--num_worker',
        type=int,
        default=10,
        help='num workers to preprocess')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    label_file = args.label_file
    results_file = args.results_file
    process_num = args.num_worker
    #data_root_dir = args.data_root_dir
    out_root_dir = args.out_root_dir
    if not os.path.exists(out_root_dir):
        os.makedirs(out_root_dir)

    results = np.load(results_file, allow_pickle=True)
    video_infos = np.load(label_file, allow_pickle=True)

    def data_deal(video_infos, preds, test_one, lock, counter, total_length):
        for vinfo, pred in zip(video_infos, preds):
            sample = vinfo[0] if test_one else vinfo
            # check if correct
            if pred != sample['label']:
                # get video path
                data_video_path = sample['frame_dir']
                out_video_path = os.path.join(out_root_dir, str(sample['label']), '/'.join(data_video_path.split('/')[-3:]))
                if not os.path.exists(out_video_path):
                    os.makedirs(out_video_path)
                    # copy all images to video dir
                    for img in sample['facerect_infos']:
                        src_img_path = os.path.join(data_video_path, img)
                        dst_img_path = os.path.join(out_video_path, img)
                        shutil.copy(src_img_path, dst_img_path)
                # copy images
                start_index = sample['start_index']
                end_index = start_index + sample['total_frames'] - 1
                out_images_path = os.path.join(out_video_path, '{}_{}'.format(start_index, end_index))
                if not os.path.exists(out_images_path):
                    os.makedirs(out_images_path)
                else:
                    continue
                for i in range(start_index, end_index + 1):
                    img_name = 'img_{:05d}.jpg'.format(i)
                    src_img_path = os.path.join(data_video_path, img_name)
                    dst_img_path = os.path.join(out_images_path, img_name)
                    shutil.copy(src_img_path, dst_img_path)
            # counter
            lock.acquire()
            try:
                # p_bar.update(1)
                counter.value += 1
                if counter.value % 50 == 0:
                    print(f"{counter.value}/{total_length} done.")
            finally:
                lock.release()

    def error_ana(results, video_infos, test_one=True):
        # error anasys
        assert len(results) == len(video_infos), 'results length({}) != video_infos length({})'.format(len(results),
                                                                                                       len(video_infos))
        preds = np.argmax(results, axis=1)

        files = video_infos
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
            pw = Process(target=data_deal,
                         args=(files[start_index:end_index], preds[start_index:end_index], test_one, lock, counter, len(files)))
            pw.start()
            process_pool.append(pw)

        for p in process_pool:
            p.join()





    error_ana(results, video_infos)