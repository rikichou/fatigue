import os
import json

json_file = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20210824_1.1.2_20211116.json'
video_root_dir = '/zhourui/workspace/pro/fatigue/data/rawframes/new_clean/fatigue_clips'

count = 0
with open(json_file, 'r') as fp:
    anns = json.load(fp)

    for idx,vname in enumerate(anns):
        vpath = os.path.join(video_root_dir, vname)
        if not os.path.exists(vpath):
            count += 1
            print(vpath)
        if idx%1000 == 0:
            print("{}/{}".format(idx, len(anns)))

print("Invalid ", count)