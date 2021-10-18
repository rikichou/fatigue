import json

#JSON_FILE = '/Users/zhourui/Downloads/test.json'
#JSON_FILE = '/Users/zhourui/Downloads/test_merge.json'
JSON_FILE = '/Users/zhourui/Downloads/20210824_fatigue_pl_less_than_50_fatigue_full_info_all_path.json'

with open(JSON_FILE, 'r') as f:
    infos = json.load(f)

train = 0
valid = 0
label = {}

for vname in infos:
    info = infos[vname]
    if 'train' == info['license_plate_type']:
        train += 1
    else:
        valid += 1

    label_name = info['label']
    if label_name not in label:
        label[label_name] = 0
    label[label_name] += 1

print("Total {}, train {}, valid {}".format(len(infos), train, valid))
print(label)
