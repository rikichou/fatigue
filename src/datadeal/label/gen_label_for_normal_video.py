import json
import os
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='get label for normal video')
    parser.add_argument('src_video_root_dir', type=str, help='source video directory')
    parser.add_argument('out_json_path', type=str, help='output json file path')
    parser.add_argument('label_string', type=str, help='label string')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2, 3],
        default=1,
        help='directory level of data')
    parser.add_argument(
        '--clips_per_video',
        type=int,
        default=5,
        help='directory level of data')

    args = parser.parse_args()

    return args

args = parse_args()
if __name__ == '__main__':
    normal_json_info = {}
    videos_dirs = glob.glob(args.src_video_root_dir + '/*')
    prefix = os.path.basename(args.src_video_root_dir)
    for v in videos_dirs:
        if not os.path.isdir(v):
            continue
        # get vname as dict keys
        vname = os.path.join(prefix, os.path.basename(v))
        print(prefix, v, os.path.basename(v))

        video_info = {}
        # train or valid
        video_info['license_plate_type'] = 'train'

        # total frames
        imgs = glob.glob(os.path.join(v, '*.jpg'))
        video_info['frames_avi'] = len(imgs)

        # add label
        video_info['label'] = args.label_string

        # get fatigue warning idx(end index)
        total_frames = len(imgs)
        clip_len = int(total_frames / args.clips_per_video)

        video_info['fatigue_warning_idx'] = list(range(clip_len, total_frames, clip_len))

        # add to total info
        normal_json_info[vname] = video_info

    with open(args.out_json_path, 'w') as f:
        json.dump(normal_json_info, f)


