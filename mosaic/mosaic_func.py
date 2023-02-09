from my_utils import IMAGE2, GenExtents, coord_ras2geo, coord_geo2ras, make_file, run_one_erase
from easy_mosaic import raster_mosaic
from osgeo import gdal, ogr
import numpy as np
from numpy import einsum
import numexpr as ne
import os
import cv2
import sys
from time import time
t0 = time()
img_list = [
    r'C:\Users\DELL\Desktop\l123\l1_8bit.tif',
    r'C:\Users\DELL\Desktop\l123\l2_8bit.tif',
]
output_ol_path = r'C:\Users\DELL\Desktop\l123\tmp\overlap.tif'
output_path = r'C:\Users\DELL\Desktop\l123\tmp\mosaic3.tif'

img1 = IMAGE2()
img2 = IMAGE2()
img1.read_img(img_list[0])
img2.read_img(img_list[1])

print(img1.im_geotrans)
print(img2.im_geotrans)
x1_res, y1_res = img1.im_geotrans[1], img1.im_geotrans[5]
x2_res, y2_res = img2.im_geotrans[1], img2.im_geotrans[5]

# 输出参数
# # 左上角坐标
lx1_geo = img1.im_geotrans[0]
ly1_geo = img1.im_geotrans[3]
lx2_geo = img2.im_geotrans[0]
ly2_geo = img2.im_geotrans[3]
lx_geo, ly_geo = min(lx1_geo, lx2_geo), max(ly1_geo, ly2_geo)

# 右下角坐标
rx1_geo = lx1_geo + img1.im_width * x1_res
ry1_geo = ly1_geo + img1.im_height * y1_res
rx2_geo = lx2_geo + img2.im_width * x2_res
ry2_geo = ly2_geo + img2.im_height * y2_res
rx_geo, ry_geo = max(rx1_geo, rx2_geo), min(ry1_geo, ry2_geo)

# 获取重叠区的四至范围
lx0, ly0 = max(lx1_geo, lx2_geo), min(ly1_geo, ly2_geo)
rx0, ry0 = min(rx1_geo, rx2_geo), max(ry1_geo, ry2_geo)
x_res, y_res = x1_res, y2_res  # 分辨率
geotrans = (lx0, x_res, 0.0, ly0, 0.0, y_res)
proj = img1.im_proj
# width = int((rx_geo - lx_geo) / x_res)
# height = int((ry_geo - ly_geo) / y_res)

# 重叠区影像栅格extent
x1, y1 = coord_geo2ras(img1.im_geotrans, [lx0, ly0])
x2, y2 = coord_geo2ras(img2.im_geotrans, [lx0, ly0])

if x1 + y1 != 0:
    width_ol = img1.im_width - x1
    height_ol = img1.im_height - y1
else:
    width_ol = img2.im_width - x2
    height_ol = img2.im_height - y2

extent1_ol = [x1, y1, width_ol, height_ol]
extent2_ol = [x2, y2, width_ol, height_ol]
extent_mosaic = []

ex1 = img1.get_extent(extent1_ol)
ex2 = img2.get_extent(extent2_ol)


weight_map = (np.ones([height_ol, width_ol], dtype=float))
weight_map[ex1[0, :, :] == 0] = 0
weight_map[ex2[0, :, :] == 0] = 0

kener_size = min(height_ol, width_ol) // 2
kernel = np.ones((kener_size, kener_size), dtype=float) / (kener_size * kener_size * 1.0)
weight_map = cv2.erode(weight_map, kernel=kernel, iterations=1)
weight_map = cv2.filter2D(weight_map, -1, kernel, borderType=cv2.BORDER_CONSTANT)

# weight_map = cv2.resize(
#     weight_map, [width_ol, height_ol], interpolation=cv2.INTER_LINEAR
# )
# weight_map = (weight_map - np.min(weight_map)) / (np.max(weight_map) - np.min(weight_map))

weight_map = np.stack([weight_map, weight_map, weight_map], axis=0)

out = weight_map * ex1 + (1 - weight_map) * ex2
out[ex1 == 0] = 0
out[ex2 == 0] = 0

# 创建文件
driver = gdal.GetDriverByName("GTiff")
ds_mosaic = driver.Create(output_ol_path, width_ol, height_ol, 3, gdal.GDT_Byte)
ds_mosaic.SetGeoTransform(geotrans)
ds_mosaic.SetProjection(proj)

for i in range(3):
    ds_mosaic.GetRasterBand(i + 1).WriteArray(out[i], xoff=0, yoff=0)
del ds_mosaic

raster_mosaic(
    file_path_list=img_list + [output_ol_path],
    output_path=output_path,
)

print(time() - t0)