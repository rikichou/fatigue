import os
import glob
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory')
    parser.add_argument(
        '--level',
        type=int,
        default=3,
        choices=[1, 2, 3],
        help='directory level of data')
    parser.add_argument(
        '--is_dir',
        type=int,
        default=1,
        help='dir or avi')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # get all video folders
    files = glob.glob(args.src_folder + '/*'*args.level)
    # statistics
    sta = {}
    for f in files:
        cat_name = f.rsplit('/', maxsplit=2)[1]
        if cat_name not in sta:
            sta[cat_name] = 0
        if args.is_dir and os.path.isdir(f):
            sta[cat_name] += 1
        elif not args.is_dir:
            _,ext = os.path.splitext(f)
            print(ext)
            if ext.lower() in ['.avi', '.mp4']:
                sta[cat_name] += 1
    print(sta)



if __name__ == '__main__':
    main()