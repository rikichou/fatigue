import os
import argparse
import glob
import json
import os.path as osp
import random
import fnmatch

#label_map = {'fatigue_close':1, 'fatigue_look_down':0, 'others':2, 'yawn':2}
label_map = {'fatigue_close':1, 'fatigue_look_down':0}

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder_frames', type=str, help='root directory for the frames')
    parser.add_argument(
        'src_folder_aitxt', type=str, help='root directory for the aitxt')
    parser.add_argument(
        'out_dir', type=str, help='root directory for the aitxt')
    parser.add_argument(
        '--dataset', type=str, default='streamax', help='dataset name')
    parser.add_argument(
        '--rgb-prefix', type=str, default='img_', help='prefix of rgb frames')
    parser.add_argument(
        '--subset',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='subset to generate file list')
    parser.add_argument(
        '--level',
        type=int,
        default=3,
        choices=[1, 2, 3],
        help='directory level of data')
    parser.add_argument(
        '--out-root-path',
        type=str,
        default='data/',
        help='root path for output')
    parser.add_argument(
        '--output-format',
        type=str,
        default='txt',
        choices=['txt', 'json'],
        help='built file list format')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=False,
        help='whether to shuffle the file list')
    args = parser.parse_args()

    return args


def get_fatigue_index_from_aitxt(filepath):
    """
    解析Aitext记录的信息
    :param filepath: (str) file path
    :return
        ret_list: (list) fatigue warning frame index
    """
    ret_list = []
    with open(filepath, "r") as fp:
        frame_idx = -1
        for txt_line in fp.readlines():
            txt_line = txt_line.strip()
            if "frameindexinmp4:" in txt_line:
                frame_idx = int(txt_line.split(":")[1])
            elif "grp9:100,200:FATIGUE" in txt_line or "grp1:100,200:FATIGUE" in txt_line:
                if frame_idx == -1:
                    continue
                ret_list.append(frame_idx)  # 记录疲劳报警是视频中的第几帧
                frame_idx = -1

    return  ret_list

def build_file_list(src_folder_aitxt, frame_info, shuffle=False):
    """Build file list for a certain data split.

    Args:
        src_folder_aitxt (str): aitxt files root dir
        frame_info (dict): Dict mapping from frames to path. e.g.,
            'Skiing/v_Skiing_g18_c02': ('data/ucf101/rawframes/Skiing/v_Skiing_g18_c02', 0, 0).  # noqa: E501
        shuffle (bool): Whether to shuffle the file list.

    Returns:
        tuple: RGB file list for training and testing, together with
            Flow file list for training and testing.
    """

    def build_list():
        """Build RGB and Flow file list with a given split.

        Args:
            split (list): Split to be generate file list.

        Returns:
            tuple[list, list]: (rgb_list, flow_list), rgb_list is the
                generated file list for rgb, flow_list is the generated
                file list for flow.
        """
        invalid_count = 0
        statistics_info = {}

        statistics_info['total'] = len(frame_info)
        statistics_info['no_fatigue_warning'] = []
        statistics_info['no_aitxt'] = []
        statistics_info[0] = 0
        statistics_info[1] = 0
        statistics_info[2] = 0

        rgb_list, flow_list = list(), list()
        for item in frame_info:
            video_prefix = item

            # get fatigue index in video
            video_aitxt_path = os.path.join(src_folder_aitxt, video_prefix+'.aitxt')
            if not os.path.exists(video_aitxt_path):
                #print("video aitxt is not found! {}".format(video_aitxt_path))
                statistics_info['no_aitxt'].append(video_aitxt_path)
                continue
            fatigue_idx_list = get_fatigue_index_from_aitxt(video_aitxt_path)
            if len(fatigue_idx_list) < 1:
                #print("Have no Fatigue warning in {}".format(video_aitxt_path))
                statistics_info['no_fatigue_warning'].append(video_aitxt_path)
                continue

            # video_prefix, total_frame_num, label, face_sx face_sy face_ex face_ey, fatigue indexes
            rgb_list.append('{},{},{},{}\n'.format(item, frame_info[item][1], frame_info[item][2], ','.join(str(x+1) for x in fatigue_idx_list)))
            statistics_info[frame_info[item][2]] += 1

        if shuffle:
            random.shuffle(rgb_list)
        print("Statistics : \nTotal video {}\nNo fatigue warning {}\nNo aitxt {}\nlook_down {}\nclose {}\nother {}\n".format(
            statistics_info['total'],
            len(statistics_info['no_fatigue_warning']),
            len(statistics_info['no_aitxt']),
            statistics_info[0],
            statistics_info[1],
            statistics_info[2]
        ))

        return rgb_list

    train_rgb_list = build_list()

    return train_rgb_list

def parse_directory(path,
                    rgb_prefix='img_',
                    flow_x_prefix='flow_x_',
                    flow_y_prefix='flow_y_',
                    level=2):
    """Parse directories holding extracted frames from standard benchmarks.

    Args:
        path (str): Directory path to parse frames.
        rgb_prefix (str): Prefix of generated rgb frames name.
            default: 'img_'.
        flow_x_prefix (str): Prefix of generated flow x name.
            default: `flow_x_`.
        flow_y_prefix (str): Prefix of generated flow y name.
            default: `flow_y_`.
        level (int): Directory level for glob searching. Options are 1 and 2.
            default: 1.

    Returns:
        dict: frame info dict with video id as key and tuple(path(str),
            rgb_num(int), flow_x_num(int)) as value.
    """
    print(f'parse frames under directory {path}')
    if level == 1:
        # Only search for one-level directory
        def locate_directory(x):
            return osp.basename(x)

        frame_dirs = glob.glob(osp.join(path, '*'))

    elif level == 2:
        # search for two-level directory
        def locate_directory(x):
            return osp.join(osp.basename(osp.dirname(x)), osp.basename(x))

        frame_dirs = glob.glob(osp.join(path, '*', '*'))
    elif level == 3:
        # search for three-level directory
        def locate_directory(x):
            return osp.join(osp.basename(osp.dirname(osp.dirname(x))), osp.basename(osp.dirname(x)), osp.basename(x))

        frame_dirs = glob.glob(osp.join(path, '*', '*', '*'))
    else:
        raise ValueError('level can be only 1 or 2 or 3')

    def count_files(directory, prefix_list):
        """Count file number with a given directory and prefix.

        Args:
            directory (str): Data directory to be search.
            prefix_list (list): List or prefix.

        Returns:
            list (int): Number list of the file with the prefix.
        """
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x + '*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, frame_dir in enumerate(frame_dirs):
        label_str = os.path.basename(os.path.dirname(frame_dir))
        if label_str not in label_map:
            print("{} not label map!!".format(label_str))
            continue
        total_num = count_files(frame_dir,
                                (rgb_prefix, 'hahahah', 'hohhohoh'))
        dir_name = locate_directory(frame_dir)
        label = label_map[label_str]

        if i % 10000 == 0:
            print(f'{i} videos parsed')
        frame_dict[dir_name] = (frame_dir, total_num[0], label)

    print('frame directory analysis done')
    return frame_dict

def main():
    args = parse_args()

    # get frames info
    frame_info = parse_directory(
        args.src_folder_frames,
        rgb_prefix=args.rgb_prefix,
        level=args.level)

    out_path = args.out_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    lists = build_file_list(args.src_folder_aitxt, frame_info, shuffle=args.shuffle)

    filename = f'{args.dataset}_{args.subset}_list_rawframes.txt'
    with open(osp.join(out_path, filename), 'w') as f:
        f.writelines(lists)


if __name__ == '__main__':
    main()
