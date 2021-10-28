import os
import json

ann_file = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20211028_fatigue_lookdown_squint_calling_smoking.json'
video_data_prefix = '/zhourui/workspace/pro/fatigue/data/rawframes/new_clean'
facerect_data_prefix = '/zhourui/workspace/pro/fatigue/data/anns/new_clean'

statistics = {}
with open(ann_file, 'r') as fp:
    anns = json.load(fp)

    for idx,vname in enumerate(anns):
        vinfo = anns[vname]

        # video path
        video_path = os.path.join(video_data_prefix, vname)
        if not os.path.exists(video_path):
            print("Video not found ", video_path)
            continue

        # video face rectangle
        facerect_path = os.path.join(facerect_data_prefix, vname + '.json')
        if not os.path.exists(facerect_path):
            print("rect json not found ", facerect_path)
            continue

        # video label
        label = vinfo['label']
        if label not in statistics:
            statistics[label] = 0
        statistics[label] += 1

        if idx%1000 == 0:
            print("{}/{}".format(idx, len(anns)))

