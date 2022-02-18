import os
import glob

from matplotlib import image

IMAGE_DIR = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images_20211227/test_1227bins'
out_file_path = '/home/ruiming/workspace/pro/fatigue/data/lianyong/face_video_images_20211227/test_1227bins/imglist.txt'

images_list = glob.glob(os.path.join(IMAGE_DIR, '*.bin'))
with open(out_file_path,'w') as fp:
    for imgpath in images_list:
        fp.write(os.path.basename(imgpath)+'\n')