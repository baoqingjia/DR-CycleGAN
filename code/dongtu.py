import cv2
import os
import glob
import numpy as np
import scipy.io as sio

im_dir = '/home/user/data4/chongxin/fastdvdnet_off/output_ori/moni/check1/'  # 图片存储路径
video_dir = '/home/user/data4/chongxin/fastdvdnet_off/output_ori/moni/motion.avi' # 视频存储路径及视频名

# seqs_dirs = sorted(glob.glob(os.path.join(im_dir, '*')))
# num = len(seqs_dirs)
# imags = np.zeros((num,256,960),dtype=np.uint8)
# i = 0
#
# for file_path in seqs_dirs:
#     file = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
#     imags[i,:,:] = file[:,:]
#     i = i + 1

data = sio.loadmat('/home/user/data4/chongxin/fastdvdnet_off/output_ori/moni/motion.mat')
imags = data['motion']

fps = 2 # 帧率一般选择20-30

_, height, width = imags.shape

video_handler = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height), isColor=False)
for i, img in enumerate(imags):
    img = np.uint8(img).copy()
    img = cv2.putText(img, "{:03d}".format(i+1), (20, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, 255, thickness=2)
    video_handler.write(img)
cv2.destroyAllWindows()
video_handler.release()
