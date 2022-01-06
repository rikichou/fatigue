import sys
import os
import shutil
import glob
import cv2

imgs_root_dir = r'E:\software\nvtai_tool\config\fatigue_two_stage\cnn25\data\face_video_images\test\UN\zheA9S38000000000-210116-130009-130109-01p016000000_FATIGUE_130032_130052'
out_root_dir = r'E:\software\nvtai_tool\config\fatigue_two_stage\cnn25\data\resized_face_video_images\test\UN\zheA9S38000000000-210116-130009-130109-01p016000000_FATIGUE_130032_130052'
if not os.path.exists(out_root_dir):
    os.makedirs(out_root_dir)

#imgs = glob.glob(os.path.join(imgs_root_dir, '*.jpg'))
imgs = os.listdir(imgs_root_dir)

imgs.sort()

print(imgs)
for i,img in enumerate(imgs):
    
    src_img_path = os.path.join(imgs_root_dir, img)
    image = cv2.imread(src_img_path)
    image = cv2.resize(image, (224, 224))

    out_path = os.path.join(out_root_dir, '{}.jpg'.format(i))
    cv2.imwrite(out_path, image)

# for img in imgs:
#     image = cv2.imread(img)
#     image = cv2.resize(image, (224, 224))

#     out_path = os.path.join(out_root_dir, os.path.basename(img))
#     cv2.imwrite(out_path, image)