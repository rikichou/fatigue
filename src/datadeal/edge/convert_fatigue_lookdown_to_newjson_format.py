import os
import json

json_file = '/Users/zhourui/workspace/pro/fatigue/data/cleaned_json/20210824_1.1.2_20211116.json'
old_json_file = '/Users/zhourui/workspace/pro/fatigue/data/cleaned_json/20210824_fatigue_lookdown.json'

out_json_file = '/Users/zhourui/workspace/pro/fatigue/data/cleaned_json/edge/20211122_fatigue_lookdown.json'

prefix = 'fatigue_clips'

def get_fatigue_warning_index_and_type(ann):
    idxs = ann['fatigue_warning_idx']
    label_str = ann['label']

    if len(idxs)!=len(label_str):
        return None
    assert len(idxs)==len(label_str), "len(idxs) {} != len(label_str) {}, {}".format(len(idxs), len(label_str), ann)

    out_dict = {}

    for i,idx in enumerate(idxs):
        l_str = label_str[i]
        # new label
        if l_str not in out_dict:
            out_dict[l_str] = []
        out_dict[l_str].append(idx)

    return out_dict


with open(old_json_file, 'r') as fp:
    old_anns = json.load(fp)

invalid_count = 0
empty_count = 0
c = 0
out_json_info = []
with open(json_file, 'r') as fp:
    anns = json.load(fp)

    for idx,vname in enumerate(anns):
        ann = anns[vname]
        label_index_and_type = get_fatigue_warning_index_and_type(ann)

        if label_index_and_type is None:
            invalid_count += 1
            continue

        if len(label_index_and_type) < 1:
            empty_count += 1
            continue

        for label_str in label_index_and_type:
            info = {}
            info['vname'] = os.path.join(prefix, vname)
            info['license_plate_type'] = ann['license_plate_type']
            info['frames_avi'] = old_anns[info['vname']]['frames_avi']
            info['label'] = label_str
            info['fatigue_warning_idx'] = label_index_and_type[label_str]

            out_json_info.append(info)

        if idx%1000 == 0:
            print("{}/{}".format(idx, len(anns)))

with open(out_json_file, 'w') as f:
    json.dump(out_json_info, f)
print("Total {}, Invalid {}, Empty {}, out {}".format(len(anns), invalid_count, empty_count, len(out_json_info)))