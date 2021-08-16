import os
import glob

def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False

IMGS_ROOT_DIR = '/zhourui/workspace/pro/fatigue/data/video/train/fatigue_20210122_20210131'

level = 2
videos = glob.glob(os.path.join(IMGS_ROOT_DIR, '/*' * level + '.avi'))

for v in videos:
    if is_contains_chinese(v):
        print(v)
        break



