import os
import json

json_file = '/Users/zhourui/workspace/pro/fatigue/data/cleaned_json/20211028_calling_smoking.json'
out_json_file = '/Users/zhourui/workspace/pro/fatigue/data/cleaned_json/edge/20211122_calling_smoking.json'

invalid_count = 0
empty_count = 0
c = 0
out_json_info = []
with open(json_file, 'r') as fp:
    anns = json.load(fp)

    for idx,vname in enumerate(anns):
        ann = anns[vname]

        info = {}
        info['vname'] = vname
        info['license_plate_type'] = ann['license_plate_type']
        info['frames_avi'] = ann['frames_avi']
        info['label'] = ann['label']
        info['fatigue_warning_idx'] = ann['fatigue_warning_idx']

        out_json_info.append(info)

        if idx%1000 == 0:
            print("{}/{}".format(idx, len(anns)))

with open(out_json_file, 'w') as f:
    json.dump(out_json_info, f)
print("Total {}, Invalid {}, Empty {}, out {}".format(len(anns), invalid_count, empty_count, len(out_json_info)))