import os
import glob

IMAGE_DIR = r'E:\software\nvtai_tool\config\fatigue_two_stage\cnn25\data\lianyongimages'
out_file_path = r'E:\software\nvtai_tool\config\fatigue_two_stage\cnn25\data\lianyongimages\image_list_all_with_prefix.txt'

images_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))

with open(out_file_path,'w') as fp:
    for imgpath in images_list:
        fp.write(imgpath+'\n')