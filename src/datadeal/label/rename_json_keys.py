import os
import json

jsonfile = '/zhourui/workspace/pro/fatigue/data/clean/fatigue_anns/20210824_fatigue_pl_less_than_50_fatigue_full_info_all_path.json'
outjsonfile = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20210824_fatigue_lookdown.json'

prefix = 'fatigue_clips'

with open(jsonfile, 'r') as fp:
    anns = json.load(fp)

outanns = {}
for k in anns:
    newk = os.path.join(prefix, k)
    outanns[newk] = anns[k]

with open(outjsonfile, 'w') as f:
    json.dump(outanns, f)