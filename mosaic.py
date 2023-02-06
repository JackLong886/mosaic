from osgeo import gdal, gdalconst
import os
import argparse
import json
import sys
import time
import cv2
from numpy import einsum
import numpy as np
import shutil

t0 = time.time()


class IMAGE2:
    # 读图像文件
    def read_img(self, filename, ):
        self.in_file = filename
        self.dataset = gdal.Open(self.in_file)  # 打开文件
        self.im_width = self.dataset.RasterXSize  # 栅格矩阵的列数
        self.im_height = self.dataset.RasterYSize  # 栅格矩阵的行数
        self.im_bands = self.dataset.RasterCount  # 波段数
        self.im_geotrans = self.dataset.GetGeoTransform()  # 仿射矩阵，左上角像素的大地坐标和像素分辨率
        self.im_proj = self.dataset.GetProjection()  # 地图投影信息，字符串表示
        del self.dataset

    def get_extent(self, extent):
        x, y, s_size, y_size = extent
        dataset = gdal.Open(self.in_file)
        extent_img = dataset.ReadAsArray(x, y, s_size, y_size)
        return extent_img

    def create_img(self, filename, out_bands=0, im_width=0, im_height=0, im_proj=0, im_geotrans=0,
                   datatype=gdal.GDT_Byte):
        self.datatype = datatype
        if out_bands == 0:
            self.out_bands = self.im_bands
        else:
            self.out_bands = out_bands
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        if im_width != 0 and im_height != 0:
            self.output_dataset = driver.Create(filename, im_width, im_height, out_bands, datatype)
        else:
            self.output_dataset = driver.Create(filename, self.im_width, self.im_height, self.out_bands, self.datatype)

        if im_geotrans != 0:
            self.output_dataset.SetGeoTransform(im_geotrans)
        else:
            self.output_dataset.SetGeoTransform(self.im_geotrans)  # 写入仿射变换参数
        if im_proj != 0:
            self.output_dataset.SetProjection(im_proj)
        else:
            self.output_dataset.SetProjection(self.im_proj)  # 写入投影

    def write_extent(self, extent, im_data):
        x, y, s_size, y_size = extent
        if self.out_bands == 1:
            self.output_dataset.GetRasterBand(1).WriteArray(im_data, xoff=x, yoff=y)  # 写入数组数据
        else:
            for i in range(self.out_bands):
                self.output_dataset.GetRasterBand(i + 1).WriteArray(im_data[i], xoff=x, yoff=y)

    def compute_statistics(self):
        # min max mean std
        statis = []
        for i in range(self.im_bands):
            datasets = gdal.Open(self.in_file)
            s = datasets.GetRasterBand(i + 1).ComputeStatistics(True)
            statis.append(s)
        return statis

    def copy_image(self, filename):
        dirname = os.path.dirname(filename)
        make_file(dirname)
        self.copy_image_file = filename
        # 判断栅格数据的数据类型
        self.dataset = gdal.Open(self.in_file)
        im_data = self.dataset.ReadAsArray(0, 0, self.im_width, self.im_height)
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        self.copy_dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        self.copy_dataset.SetGeoTransform(self.im_geotrans)  # 写入仿射变换参数
        self.copy_dataset.SetProjection(self.im_proj)  # 写入投影
        if im_bands == 1:
            self.copy_dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                self.copy_dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

        del self.dataset

    def write2copy_image(self, extent, im_data):
        # dataset = gdal.Open(self.copy_image_file, gdal.GA_Update)
        dataset = self.copy_dataset
        x, y, s_size, y_size = extent
        bands = dataset.RasterCount
        if bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data, xoff=x, yoff=y)  # 写入数组数据
        else:
            for i in range(bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i], xoff=x, yoff=y)


def _scale_array(arr, clip=True):
    """
    Trim NumPy array values to be in [0, 255] range with option of
    clipping or scaling.

    Parameters:
    -------
    arr: array to be trimmed to [0, 255] range
    clip: should array be scaled by np.clip? if False then input
        array will be min-max scaled to range
        [max([arr.min(), 0]), min([arr.max(), 255])]

    Returns:
    -------
    NumPy array that has been scaled to be in [0, 255] range
    """
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)

    return scaled


def make_file(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _min_max_scale(arr, new_range=(0, 255)):
    """
    Perform min-max scaling to a NumPy array

    Parameters:
    -------
    arr: NumPy array to be scaled to [new_min, new_max] range
    new_range: tuple of form (min, max) specifying range of
        transformed array

    Returns:
    -------
    NumPy array that has been scaled to be in
    [new_range[0], new_range[1]] range
    """
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled


def coord_ras2geo(im_geotrans, coord):
    x0, x_res, _, y0, _, y_res = im_geotrans
    x = x0 + x_res * coord[0]
    y = y0 + y_res * coord[1]
    return x, y


def coord_geo2ras(im_geotrans, coord):
    x0, x_res, _, y0, _, y_res = im_geotrans
    x = int(round(abs((coord[0] - x0) / x_res)))
    y = int(round(abs((coord[1] - y0) / y_res)))
    return x, y


def callback(v1, v2, v3):
    sys.stdout.flush()
    print("mosaic:{:.4f}".format(v1), flush=True)


def raster_mosaic(file_path_list, output_path):
    print("raster mosaic")
    assert len(file_path_list) > 1
    ds_list = []
    reference_file_path = file_path_list[0]
    input_file1 = gdal.Open(reference_file_path, gdal.GA_ReadOnly)
    input_proj1 = input_file1.GetProjection()

    for path in file_path_list:
        ds = gdal.Open(path, gdal.GA_ReadOnly)
        ds_list.append(ds)

    options = gdal.WarpOptions(
        # srcSRS=input_proj1,
        # dstSRS=input_proj1,
        format='GTiff',
        srcNodata=0,
        dstNodata=0,
        resampleAlg=gdalconst.GRA_Bilinear,
        callback=callback
    )
    gdal.Warp(output_path, ds_list, options=options)


def color_transfer_para(input, img_stats, ref_stats, clip=True, preserve_paper=True):
    trans = False
    if input.shape[0] == 3:
        trans = True
        input = einsum('ijk->jki', input)
    target = cv2.cvtColor(input, cv2.COLOR_BGR2LAB).astype("float32")

    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = ref_stats
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = img_stats

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    if preserve_paper:
        # scale by the standard deviations using paper proposed factor
        l *= (lStdTar / lStdSrc)
        a *= (aStdTar / aStdSrc)
        b *= (bStdTar / bStdSrc)
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        l *= (lStdSrc / lStdTar)
        a *= (aStdSrc / aStdTar)
        b *= (bStdSrc / bStdTar)
    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall
    # outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    transfer[input == 0] = 0
    if trans:
        transfer = einsum('jki->ijk', transfer)
    # return the color transferred image
    return transfer


def GenExtents(width, height, win_size, win_std=0):
    if win_std == 0:
        win_std = win_size
    frame = []
    x = 0
    y = 0
    while y < height:  # 高度方向滑窗
        if y + win_size >= height:
            y_left = height - win_size
            y_right = win_size
            y_end = True
        else:
            y_left = y
            y_right = win_size
            y_end = False

        while x < width:  # 宽度方向滑窗
            if x + win_size >= width:
                x_left = width - win_size
                x_right = win_size
                x_end = True
            else:
                x_left = x
                x_right = win_size
                x_end = False
            frame.append((x_left, y_left, x_right, y_right))
            x += win_std
            if x_end:
                break
        y += win_std
        x = 0
        if y_end:
            break
    return frame


def image_stats_path(image_path):
    # compute the mean and standard deviation of each channel
    image = cv2.imread(image_path)[:, :, (2, 1, 0)]
    (lMean, lStd, aMean, aStd, bMean, bStd) = image_stats(image)

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def image_stats(image):
    # compute the mean and standard deviation of each channel
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def generate_overview(img_path, work_dir, pct=5):
    basename = os.path.basename(img_path)
    preffix, _ = os.path.splitext(basename)
    make_file(work_dir)
    out_path = os.path.join(work_dir, preffix + '_overview.jpg')
    options = gdal.TranslateOptions(
        format='JPEG',
        # width=500, height=500
        heightPct=pct, widthPct=pct
    )
    gdal.Translate(
        destName=out_path,
        srcDS=img_path,
        options=options
    )
    return out_path


def run_one_color(input_path, ref_path, output_path):
    in_img = IMAGE2()
    ref_img = IMAGE2()

    in_img.read_img(input_path)
    ref_img.read_img(ref_path)
    in_img.copy_image(output_path)

    in_stat = image_stats_path(generate_overview(input_path, opt.work_dir, pct=5))
    ref_stat = image_stats_path(generate_overview(ref_path, opt.work_dir, pct=5))

    if in_img.im_width * in_img.im_height > 10240 * 10240:
        extents = GenExtents(in_img.im_width, in_img.im_height, win_size=2048)
        for i, extent in enumerate(extents):
            in_patch = in_img.get_extent(extent)
            out_patch = color_transfer_para(
                input=in_patch,
                img_stats=in_stat,
                ref_stats=ref_stat,
                clip=True,
                preserve_paper=False
            )
            # 写出
            in_img.write2copy_image(extent=extent, im_data=out_patch)
            # bg.write_extent(extent=extent_bg, im_data=out_patch)

    else:
        extent = [0, 0, in_img.im_width, in_img.im_height]
        in_patch = in_img.get_extent(extent)
        out_patch = color_transfer_para(
            input=in_patch,
            img_stats=in_stat,
            ref_stats=ref_stat,
            clip=True,
            preserve_paper=False
        )
        # 写出
        in_img.write2copy_image(extent=extent, im_data=out_patch)


def crop_img(image_path_list, shp_path_list, output_dir):
    assert len(image_path_list) == len(shp_path_list)
    make_file(output_dir)
    output_path_list = []
    for image_path, shp_path in zip(image_path_list, shp_path_list):
        # print('start crop {} using {}'.format(image_path, shp_path))
        basename = os.path.basename(image_path)
        preffix, _ = os.path.splitext(basename)
        name = 'crop_' + preffix + '.vrt'
        output_path = os.path.join(output_dir, name)

        options = gdal.WarpOptions(
            format='VRT',
            cutlineDSName=shp_path,
            dstNodata=0,
            cropToCutline=True
        )
        datasets = gdal.Warp(
            output_path,
            image_path,
            options=options
        )
        if datasets is None:
            raise KeyError('crop error')

        output_path_list.append(output_path)
    return output_path_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cloud removal for remote sensing images')
    parser.add_argument('--json_path', type=str, default=r'parameters.json')
    parser.add_argument('--img_list', type=list, default=None)
    parser.add_argument('--shp_list', type=list, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--work_dir', type=str, default=r'C:\Users\DELL\Desktop\ceshi\tmp')
    opt = parser.parse_args()
    if os.path.exists(opt.json_path):
        with open(opt.json_path, 'r', encoding='utf-8-sig') as f:
            args = json.load(f)
            opt.img_list = args['img_list']
            opt.shp_list = args['shp_list']
            opt.output_path = args['output_path']
            opt.work_dir = args['work_dir']

    crop_img_list = crop_img(
        image_path_list=opt.img_list,
        shp_path_list=opt.shp_list,
        output_dir=opt.work_dir,
    )

    bg = crop_img_list.pop(0)
    img_color_list = []
    img_color_list.append(bg)
    for i in range(len(crop_img_list)):
        t = time.time()
        output_path = os.path.join(opt.work_dir, 'tmp{}.tif'.format(t))
        run_one_color(input_path=opt.img_list[i], ref_path=bg,
                      output_path=output_path)
        img_color_list.append(output_path)

    print(img_color_list)
    raster_mosaic(
        file_path_list=img_color_list,
        output_path=opt.output_path,
    )
    print("CostTime:{}".format(time.time() - t0), flush=True)

    shutil.rmtree(opt.work_dir)
