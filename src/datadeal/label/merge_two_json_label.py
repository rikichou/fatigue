import json

JSON1 = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20210824_fatigue_lookdown.json'
JSON2 = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20211027_squint.json'

OUT_JSON = '/zhourui/workspace/pro/fatigue/data/anns/new_clean/20211027_fatigue_lookdown_squint.json'

with open(JSON1, 'r') as f:
    json1 = json.load(f)

with open(JSON2, 'r') as f:
    json2 = json.load(f)

json1.update(json2)

with open(OUT_JSON, 'w') as f:
    json.dump(json1, f)


