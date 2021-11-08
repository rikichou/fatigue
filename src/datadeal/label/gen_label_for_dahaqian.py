import os
import numpy as np
import glob
import json

label_dir = r'H:\pro\fatigue\data\dahaqian\fatigue_dahaqian_outkeypoints_multi'
out_json_path = r'H:\pro\fatigue\data\dahaqian\20211108_fatigue_dahaqian.json'

labels = glob.glob(label_dir+r'\*.npy')
min_frames_num = 10

prefix = 'fatigue_dahaqian'
label_str = prefix

normal_json_info = {}
for count,l in enumerate(labels):
    # read label info
    info = np.load(l, allow_pickle=True).item()

    # get video
    vname = os.path.splitext(os.path.basename(l))[0]
    vpath = prefix + '/' + vname

    # get max image index
    indexs = [int(name.split('.')[0].split('_')[1]) for name in info.keys()]
    max_idx = max(indexs)

    start_idx = -1
    end_idx = -1
    valid_idxs = []
    for idx in range(max_idx):
        frame_idx = idx + 1

        # get image label
        imgname = 'img_{:05d}.jpg'.format(frame_idx)
        if imgname not in info:
            # reset idx info
            start_idx = -1
            end_idx = -1
            continue

        imginfo = info[imgname]
        if imginfo['facerect'] == None:
            # reset idx info
            start_idx = -1
            end_idx = -1
        else:
            # face rectangle
            sx, sy, ex, ey = imginfo['facerect']
            sx = int(sx)
            sy = int(sy)
            ex = int(ex)
            ey = int(ey)

            # points
            points = imginfo['points']
            l = points[60]
            r = points[64]
            t = points[62]
            b = points[66]

            w = r[0] - l[0]
            h = b[1] - t[1]

            ratio = h / (w+1e-5)
            color = (0, 255, 0)
            if ratio > 0.5:
                # update idx info
                end_idx = frame_idx
                if start_idx == -1 or (end_idx - start_idx)<min_frames_num:
                    # 1. have no enough frames
                    start_idx = frame_idx if start_idx == -1 else start_idx
                else:
                    # have enough frames
                    valid_idxs.append(frame_idx)
                    start_idx = -1
                    end_idx = -1
            else:
                # reset idx info
                start_idx = -1
                end_idx = -1

    if len(valid_idxs) > 0:
        video_info = {}
        # train or valid
        video_info['license_plate_type'] = 'train'

        # total frames
        video_info['frames_avi'] = len(indexs)

        # add label
        video_info['label'] = label_str

        # get fatigue warning idx(end index)
        video_info['fatigue_warning_idx'] = valid_idxs

        # add to total info
        normal_json_info[vpath] = video_info

    if count%1000 == 0:
        print("{}/{}".format(count, len(labels)))

print("Total {} and get {} valid dahaqian videos!".format(len(labels), len(normal_json_info)))

with open(out_json_path, 'w') as f:
    json.dump(normal_json_info, f)

