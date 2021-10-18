import json
import os
import glob

VIDEOS_ROOT_DIR = '/zhourui/workspace/pro/fatigue/data/rawframes/new_clean/fatigue_normal_squint'
OUT_JSON_PATH = '/zhourui/workspace/pro/fatigue/data/clean/fatigue_anns/20211018_fatigue_normal_squint.json'

# VIDEOS_ROOT_DIR = '/Users/zhourui/Downloads/test'
# OUT_JSON_PATH = '/Users/zhourui/Downloads/test.json'

videos_dirs = glob.glob(VIDEOS_ROOT_DIR+'/*/*')
clips_per_video = 5

normal_json_info = {}
for v in videos_dirs:
    if not os.path.isdir(v):
        continue
    # get vname as dict keys
    org_vname = '/'.join(v.rsplit('/', maxsplit=3)[-3:])
    vname = '../' + org_vname

    video_info = {}
    # train or valid
    video_info['license_plate_type'] = 'train' if 'train' in org_vname else 'valid'

    # total frames
    imgs = glob.glob(os.path.join(v, '*.jpg'))
    video_info['frames_avi'] = len(imgs)

    # add label
    video_info['label'] = 'fatigue_squint'

    # get fatigue warning idx(end index)
    total_frames = len(imgs)
    clip_len = int(total_frames / clips_per_video)

    video_info['fatigue_warning_idx'] = list(range(clip_len, total_frames, clip_len))

    # add to total info
    normal_json_info[vname] = video_info

with open(OUT_JSON_PATH, 'w') as f:
    json.dump(normal_json_info, f)


