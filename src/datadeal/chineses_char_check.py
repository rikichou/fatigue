#-*- coding:utf-8 -*-
import os
import glob

def is_contains_chinese(strs):
    for _char in strs:
        if u'\u4e00' <= _char <= u'\u9fff':
            return True
    return False

IMGS_ROOT_DIR = r'G:\pro\fatigue\data\clean\fatigue_look_down'

level = 1
videos = glob.glob(os.path.join(IMGS_ROOT_DIR, '*.mp4'))
for v in videos:
    if is_contains_chinese(os.path.basename(v)):
        print('Chineses char : ', v)

    if ' ' in v:
        print('Space in : ', v)



