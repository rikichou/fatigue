import os
import argparse
import shutil
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('label_file', type=str, help='label_file')
    parser.add_argument('results_file', type=str, help='results_file')
    parser.add_argument('data_root_dir', type=str, help='data_root_dir')
    parser.add_argument('out_root_dir', type=str, help='out_root_dir')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2, 3],
        default=3,
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
    label_file = args.label_file
    results_file = args.results_file

    data_root_dir = args.data_root_dir
    out_root_dir = args.out_root_dir
    if not os.path.exists(out_root_dir):
        os.makedirs(out_root_dir)

    results = np.load(results_file, allow_pickle=True)
    video_infos = np.load(label_file, allow_pickle=True)

    def error_ana(results, video_infos, test_one=True):
        # error anasys
        assert len(results) == len(video_infos), 'results length({}) != video_infos length({})'.format(len(results),
                                                                                                       len(video_infos))
        preds = np.argmax(results, axis=1)
        for vinfo, pred in zip(video_infos, preds):
            sample = vinfo[0] if test_one else vinfo
            # check if correct
            if pred != sample['label']:
                # get video path
                data_video_path = sample['frame_dir']
                out_video_path = os.path.join(out_root_dir, '/'.join(data_video_path.split('/')[-3:]))
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

    error_ana(results, video_infos)