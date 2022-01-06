import os
import glob

IMAGE_DIR = r'E:\software\nvtai_tool\config\fatigue_two_stage\cnn25\data\face_video_images\train_09bins'
out_file_path = r'E:\software\nvtai_tool\config\fatigue_two_stage\cnn25\data\face_video_images\train_09bins\imglist.txt'

images_list = glob.glob(os.path.join(IMAGE_DIR, '*.bin'))

with open(out_file_path,'w') as fp:
    for imgpath in images_list:
        fp.write(os.path.basename(imgpath)+'\n')