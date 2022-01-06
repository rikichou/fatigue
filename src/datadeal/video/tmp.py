import cv2
import os

from PIL import Image

# def get_fatigue_index_from_aitxt(filepath):
#     """
#     解析Aitext记录的信息
#     :param filepath: (str) file path
#     :return
#         ret_list: (list) fatigue warning frame index
#     """
#     ret_list = []
#     with open(filepath, "r") as fp:
#         frame_idx = -1
#         for txt_line in fp.readlines():
#             txt_line = txt_line.strip()
#             if "frameindexinmp4:" in txt_line:
#                 frame_idx = int(txt_line.split(":")[1])
#             elif "grp9:100,200:FATIGUE" in txt_line or "grp1:100,200:FATIGUE" in txt_line:
#                 if frame_idx == -1:
#                     continue
#                 ret_list.append(frame_idx)  # 记录疲劳报警是视频中的第几帧
#
#     return  ret_list
#
# a = 'a/b/sub/class/a.mp4'
#
#
#
#
# print(os.path.join(*(a.rsplit('/', maxsplit=3)[-3:])))
