import sys
import os
import shutil
import glob
import cv2

imgs_root_dir = r'E:\software\nvtai_tool\config\fatigue_two_stage\cnn25\data\face_video_images\test\CE\02_03_0_0_28076958'
out_root_dir = r'E:\software\nvtai_tool\config\fatigue_two_stage\cnn25\data\resized_face_video_images\test\CE\02_03_0_0_28076958'
if not os.path.exists(out_root_dir):
    os.makedirs(out_root_dir)

imgs = glob.glob(os.path.join(imgs_root_dir, '*.jpg'))

for img in imgs:
    image = cv2.imread(img)
    image = cv2.resize(image, (224, 224))

    out_path = os.path.join(out_root_dir, os.path.basename(img))
    cv2.imwrite(out_path, image)