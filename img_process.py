"""
dowmsample_image
将输入影像重采样至指定尺寸的缩略图
参数：
src_file:本地文件路径, 图像类型：png,jpeg,tif,img
out_file：输出文件路径， png,jpeg
target_w：输出图像像素宽
target_h：输出图像像素高
mode：重采样方式，center：以图像中心外扩采样, random:随机位置采样
pct：[可选]采样原图比例，默认为采样原图长宽的1/4
"""

from osgeo import gdal, gdalconst
import os
import random


def downsample_image(src_file: str,
                     out_file: str,
                     target_w: int,
                     target_h: int,
                     mode: str,
                     pct: float = 1 / 4):

    if not os.path.exists(src_file):
        raise KeyError('{}文件不存在'.format(src_file))
    _, suffix = os.path.splitext(os.path.basename(out_file))
    if suffix[1:] in ['jpg', 'JPG', 'jpeg', 'JPEG']:
        dst_format = 'JPEG'
    elif suffix[1:] in ['png', 'PNG']:
        dst_format = 'PNG'
    else:
        raise KeyError('{}不是支持的输出格式, 参考：png, jpeg'.format(suffix[1:]))
    ds = gdal.Open(src_file, gdal.GA_ReadOnly)
    im_width = ds.RasterXSize  # 栅格矩阵的列数
    im_height = ds.RasterYSize  # 栅格矩阵的行数

    ratio_h = (im_width / im_height) / (target_w / target_h)
    resized_w = int(im_width * pct)
    resized_h = int(im_height * pct * ratio_h)
    if resized_h >= im_height:
        resized_h = im_height

    # 裁剪ROI
    if mode == 'center':
        extent = [
            int((im_width - resized_w) * 0.5),
            int((im_height - resized_h) * 0.5),
            resized_w,
            resized_h
        ]
    elif mode == 'random':
        random_x = random.randint(0, im_width - resized_w)
        random_y = random.randint(0, im_height - resized_h)
        extent = [random_x, random_y, resized_w, resized_h]
    else:
        raise KeyError('参数mode={}不在选项[center, random]中'.format(mode))
    # 降采样到指定尺寸
    options = gdal.TranslateOptions(
        format=dst_format,
        width=target_w,
        height=target_h,
        srcWin=extent,
        resampleAlg=gdalconst.GRIORA_NearestNeighbour
    )

    out_ds = gdal.Translate(
        destName=out_file,
        srcDS=src_file,
        options=options
    )

    out_data = out_ds.ReadAsArray(0, 0, target_w, target_w)

    del out_ds
    del ds
    return out_data


if __name__ == '__main__':
    downsample_image(src_file=r'C:\Users\DELL\Desktop\l123\l1_8bit.tif',
                     out_file=r'C:\Users\DELL\Desktop\l123\l1_overview.jpg',
                     target_w=500,
                     target_h=500,
                     mode='center')
