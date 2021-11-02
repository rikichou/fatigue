import os
import numpy as np

#results_file_path = '/Users/zhourui/workspace/pro/fatigue/src/test/fatigue_r50_clean_fp16/results_fatigue_r50_clean.npy'
results_file_path = '/Users/zhourui/workspace/pro/fatigue/src/test/fatigue_face_video_tensorrt_python/model/fatigue_r50_clean_with_squint_smoke_call/results.npy'
results_file_path = '/Users/zhourui/workspace/pro/fatigue/src/test/fatigue_face_video_tensorrt_python/model/fatigue_r50_clean_withnormal/results.npy'


results = np.load(results_file_path, allow_pickle=True).item()

true_label = []
pred_label = []
video_names = []
normal_mask = np.zeros(len(results))
fatigue_mask = np.zeros(len(results))

def get_predict_label(result, threshold=0.9):
    fatigue_pred = result[:,1]
    pred_num = (fatigue_pred>threshold).sum()

    return 0 if pred_num==0 else 1

for idx,vname in enumerate(results):
    result = results[vname]

    # get predict
    pred = get_predict_label(result)
    pred_label.append(pred)

    truel = 0
    if 'error_tree' in vname:
        true_label.append(0)
        truel = 0
        normal_mask[idx] = True
        fatigue_mask[idx] = False
    else:
        true_label.append(1)
        truel = 1
        fatigue_mask[idx] = True
        normal_mask[idx] = False

    # collect vname
    video_names.append(vname)

true_label = np.array(true_label)
pred_label = np.array(pred_label)

mask = true_label==0
print("Normal total {}, Accuracy {}".format(len(true_label[mask]), (true_label[mask]==pred_label[mask]).sum() / len(true_label[mask])))

mask = true_label==1
print("Fatigue total {}, Accuracy {}".format(len(true_label[mask]), (true_label[mask]==pred_label[mask]).sum() / len(true_label[mask])))

print("total {}, Accuracy {}".format(len(true_label), (true_label==pred_label).sum() / len(true_label)))


# get all error video names
video_names = np.array(video_names)
error_pred_mask = true_label != pred_label
error_vnames = video_names[error_pred_mask]

# error statistics
error_statistics = {}
total_video_statistics = {}

for v in video_names:
    error_type = v.rsplit('/', maxsplit=2)[1]
    video_type = v.rsplit('/', maxsplit=4)[1]

    if video_type not in total_video_statistics:
        total_video_statistics[video_type] = {}

    if error_type not in total_video_statistics[video_type]:
        total_video_statistics[video_type][error_type] = []

    total_video_statistics[video_type][error_type].append(v)

for ev in error_vnames:
    # get error type
    error_type = ev.rsplit('/', maxsplit=2)[1]
    video_type = ev.rsplit('/', maxsplit=4)[1]

    if video_type not in error_statistics:
        error_statistics[video_type] = {}

    if error_type not in error_statistics[video_type]:
        error_statistics[video_type][error_type] = []

    error_statistics[video_type][error_type].append(ev)

fatigue_num = 0
normal_num = 0
for video_type in error_statistics:
    for error_type in error_statistics[video_type]:
        error_num = len(error_statistics[video_type][error_type])
        total_num = len(total_video_statistics[video_type][error_type])
        print("{}:{} Total_num {}, error_num {}, error_rate {:.2f}".format('normal' if video_type=='error_tree' else 'fatigue', error_type,
                                                                       total_num, error_num, error_num/total_num))

        if video_type == 'error_tree':
            normal_num += len(error_statistics[video_type][error_type])
        else:
            fatigue_num += len(error_statistics[video_type][error_type])
        if error_type == '疲劳闭眼':
            for name in error_statistics[video_type][error_type]:
                print(name)

print("Normal num : {}, Fatigue num : {}".format(normal_num, fatigue_num))


