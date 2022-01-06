import sys
import os

ROOT_DIR = r'G:\pro\fatigue\data\old\error_ana'

cat_statistics = {}
total = 0
for cat in os.listdir(ROOT_DIR):
    if cat=='valid':
        continue
    cat_path = os.path.join(ROOT_DIR, cat)
    if not os.path.isdir(cat_path):
        continue
    if cat not in cat_statistics:
        cat_statistics[cat] = 0
    videos = os.listdir(cat_path)
    cat_statistics[cat] = len(videos)
    total += len(videos)

print("Total ", total)
for cat in cat_statistics:
    print(cat, cat_statistics[cat])
