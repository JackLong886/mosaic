import torch
import os
import cv2
import numpy as np
import time
t0 = time.time()
img = cv2.imread(r'C:\Users\DELL\Desktop\CloudRemoval\test\ceshi_result\tmp_s.tif')  # 读取图像
dst = cv2.imread(r'C:\Users\DELL\Desktop\CloudRemoval\test\ceshi_result\tmp_bg.tif')
mask = cv2.imread(r'C:\Users\DELL\Desktop\CloudRemoval\test\ceshi_result\tmp_mask.tif')
out_path = r'C:\Users\DELL\Desktop\CloudRemoval\test\ceshi_result\tmp_cv.tif'
# cv2.imshow('temporal', img)
# cv2.imshow('cloud', dst)
# cv2.imshow('mask', mask)
# gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# # 将图片二值化
# _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# x_start = y_start = 9999
# x_end = y_end = 0
# mask_cont = np.zeros(mask.shape)
# for cont in contours:
#     # 外接矩形
#     x, y, w, h = cv2.boundingRect(cont)
#     # print(x, y, w, h)
#     x_start = np.min([x_start, x])
#     y_start = np.min([y_start, y])
#     x_end = np.max([x_end, x + w])
#     y_end = np.max([y_end, y + h])
#
#     mask_cont[y:y + h, x:x + w, :] = 255

place = np.where(mask[:, :, 0] == 255)
x_start, y_start = np.min(place[0]), np.min(place[1])
x_end, y_end = np.max(place[0]), np.max(place[1])

center = (int((y_start + y_end) / 2), int((x_start + x_end) / 2))
out = cv2.seamlessClone(src=img, dst=dst, mask=mask,
                        p=center,
                        flags=cv2.NORMAL_CLONE)

cv2.imwrite(out_path, out)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

t1 = time.time()
print(t1 - t0)
