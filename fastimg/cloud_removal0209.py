'''
输入shp改为有效区和云区
修改shp超出影像边界的错误
根据新版本的云检测程序修改代码
'''
import argparse
import json
import os
import sys
import time
import profile
import cv2
import numpy as np
from numpy import einsum
from osgeo import gdalconst, ogr, gdal
import numexpr as ne

gdal.SetConfigOption('SHAPE_ENCODING', 'gbk')
global start, end
global start2, end2


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


def make_file(path):
    if not os.path.exists(path):
        os.makedirs(path)


# 生成结合表
def union_shp(shp_path_list, img_path_list, out_dir):
    shp_num = len(shp_path_list)
    assert shp_num > 1
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset0 = driver.Open(shp_path_list[0])
    layer0 = dataset0.GetLayerByIndex(0)
    srs0 = layer0.GetSpatialRef()
    defn0 = layer0.GetLayerDefn()

    basename = os.path.basename(shp_path_list[-1])
    make_file(out_dir)
    out_path = os.path.join(out_dir, basename)

    outds = driver.CreateDataSource(out_path)
    outlayer = outds.CreateLayer(out_path, srs=srs0, geom_type=ogr.wkbPolygon)
    # 创建字段

    field_path = ogr.FieldDefn("path", ogr.OFTString)  # 创建字段(字段名，类型)
    field_path.SetWidth(100)  # 设置宽度
    outlayer.CreateField(field_path)  # 将字段设置到layer

    FieldDefns = []
    for field_n in range(defn0.GetFieldCount()):
        FieldDefns.append(defn0.GetFieldDefn(field_n))
        outlayer.CreateField(defn0.GetFieldDefn(field_n))

    for path, img_path in zip(shp_path_list, img_path_list):
        shp = driver.Open(path)
        layer = shp.GetLayerByIndex(0)
        # dir = os.path.dirname(img_path)
        layer.ResetReading()
        for feature in layer:
            geom = feature.GetGeometryRef()
            out_feat = ogr.Feature(outlayer.GetLayerDefn())
            out_feat.SetGeometry(geom)
            for FieldDefn in FieldDefns:
                out_feat.SetField(FieldDefn.GetName(), feature.GetField(FieldDefn.GetName()))
                out_feat.SetField('path', img_path)

            outlayer.CreateFeature(out_feat)
        shp.Destroy()
    dataset0.Destroy()
    outds.Destroy()


# 擦除
def erase_cloud_mask(to_erase, erase_list, erase_out_dir, inter_out_dir, valid_out_dir, image_path_list):
    make_file(erase_out_dir)
    make_file(inter_out_dir)
    # make_file(valid_out_dir)
    # 开始批量擦除
    erase_path_list = []
    inter_path_list = []
    valid_path_list = []
    new_img_path_list = []
    driver = ogr.GetDriverByName('ESRI Shapefile')
    for i, path in enumerate(erase_list):
        basename = os.path.basename(path)

        # 被擦除shp
        to_erase_shp = driver.Open(to_erase)
        to_erase_layer = to_erase_shp.GetLayer()
        num_feature = to_erase_layer.GetFeatureCount()

        if num_feature == 0:
            break

        dst_erase = os.path.join(erase_out_dir, str(i) + '_erase_' + basename)
        erase_path_list.append(dst_erase)
        dst_inter = os.path.join(inter_out_dir, str(i) + '_inter_' + basename)
        inter_path_list.append(dst_inter)

        test_shp = driver.Open(path, 0)
        test_layer = test_shp.GetLayer()
        test_srs = test_layer.GetSpatialRef()
        test_defn = test_layer.GetLayerDefn()

        outds_inter = driver.CreateDataSource(dst_inter)
        outlayer_inter = outds_inter.CreateLayer(dst_inter, srs=test_srs, geom_type=ogr.wkbPolygon)

        outds_erase = driver.CreateDataSource(dst_erase)
        outlayer_erase = outds_erase.CreateLayer(dst_erase, srs=test_srs, geom_type=ogr.wkbPolygon)

        for j in range(test_defn.GetFieldCount()):
            outlayer_inter.CreateField(test_defn.GetFieldDefn(j))
            outlayer_erase.CreateField(test_defn.GetFieldDefn(j))

        # 获取擦除剩余和擦除部分
        if i == 0:
            to_erase_layer.Erase(test_layer, outlayer_erase)
            to_erase_layer.Intersection(test_layer, outlayer_inter)

            to_erase_shp.Destroy()
        else:
            tmp_shp = driver.Open(erase_path_list[i - 1], 1)
            tmp_layer = tmp_shp.GetLayer()
            tmp_feat_count = tmp_layer.GetFeatureCount()
            if tmp_feat_count == 0:
                break
            tmp_layer.Erase(test_layer, outlayer_erase)
            tmp_layer.Intersection(test_layer, outlayer_inter)
            tmp_shp.Destroy()

        # 不相交的不输出
        if outlayer_inter.GetFeatureCount() != 0:
            new_img_path_list.append(image_path_list[i])
        else:
            inter_path_list.pop()

        # 擦除完毕
        if outlayer_erase.GetFeatureCount() == 0:
            break

        # # 添加原始影像路径字段
        # dirname = os.path.dirname(image_path_list[i])
        # outlayer_inter.CreateField(
        #     ogr.FieldDefn('Path', ogr.OFTString)
        # )
        # for feat in outlayer_inter:
        #     feat.SetField('Path', dirname)
        #     outlayer_inter.SetFeature(feat)

    return inter_path_list, new_img_path_list


# 获取云掩膜shp
def get_mask_shp(path_list, new_dir, gridcode):
    driver = ogr.GetDriverByName('ESRI Shapefile')

    new_inter_path_list = []
    make_file(new_dir)
    for path in path_list:
        shp_name = os.path.basename(path)
        new_inter_path = os.path.join(new_dir, shp_name)
        new_inter_path_list.append(new_inter_path)

        shp = driver.Open(path, 0)
        layer = shp.GetLayer()
        srs = layer.GetSpatialRef()
        defn = layer.GetLayerDefn()

        # 创建新shp
        new_inter = driver.CreateDataSource(new_inter_path)
        new_inter_layer = new_inter.CreateLayer(new_inter_path, srs=srs, geom_type=ogr.wkbPolygon)
        for j in range(defn.GetFieldCount()):
            new_inter_layer.CreateField(defn.GetFieldDefn(j))

        index = new_inter_layer.GetLayerDefn().GetFieldIndex('Shape_Area')  # 获取字段的索引值
        fld_defn = ogr.FieldDefn('Shape_Area', ogr.OFTString)  # 创建新属性的字段定义
        fld_defn.SetWidth(100)
        new_inter_layer.AlterFieldDefn(index, fld_defn, ogr.ALTER_WIDTH_PRECISION_FLAG)

        layer.ResetReading()
        for feature in layer:
            gd = feature.GetField('gridcode')
            if gd == gridcode:
                new_inter_layer.CreateFeature(feature)

    return new_inter_path_list


def shp2tif(shp_path, ref_tif_path, target_tif_path, attribute_field=''):
    ref_tif_file = IMAGE2()
    ref_tif_file.read_img(ref_tif_path)
    ref_tif_file.create_img(
        filename=target_tif_path,
        im_width=ref_tif_file.im_width, im_height=ref_tif_file.im_height,
        im_proj=ref_tif_file.im_proj, im_geotrans=ref_tif_file.im_geotrans,
        out_bands=1,
        datatype=gdal.GDT_Byte
    )

    shp_file = ogr.Open(shp_path)
    shp_layer = shp_file.GetLayer()
    gdal.RasterizeLayer(
        dataset=ref_tif_file.output_dataset,
        bands=[1],
        layer=shp_layer,
        # options=[f"ATTRIBUTE={attribute_field}"]
    )
    del ref_tif_file.output_dataset


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

    # 还没做重采样,没做滑窗


def easy_blend(bg_patch, s_patch, m_patch):
    bg = einsum('ijk->jki', bg_patch)
    s = einsum('ijk->jki', s_patch)
    if len(m_patch.shape) == 3:
        m = einsum('ijk->jki', m_patch)
    else:
        m = np.stack([m_patch, m_patch, m_patch], axis=-1)

    s = s.astype(np.float64)
    m = m.astype(np.float64) / 255.
    bg = bg.astype(np.float64)

    m2 = 1. - m
    s2 = s * m
    bg2 = bg * m2
    out = bg2 + s2

    out = einsum('ijk->kij', out)

    return out


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
        resampleAlg=gdalconst.GRA_Bilinear
    )
    gdal.Warp(output_path, ds_list, options=options)


def generate_overview(img_path, work_dir, pct=5):
    basename = os.path.basename(img_path)
    preffix, _ = os.path.splitext(basename)
    out_path = os.path.join(work_dir, 'tmp', preffix + '_overview.jpg')
    options = gdal.TranslateOptions(
        format='JPEG', resampleAlg=gdalconst.GRIORA_NearestNeighbour,
        heightPct=pct, widthPct=pct
    )
    gdal.Translate(
        destName=out_path,
        srcDS=img_path,
        options=options
    )
    return out_path


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


def cloud_removal(shp_bg, img_bg, image_path_list, shp_path_list, shp_valid_bg, shp_temp_list, opt):
    # 判断是否相交
    # 开始擦除
    inter_path_list, new_img_path_list = erase_cloud_mask(
        to_erase=shp_bg,
        erase_list=shp_path_list,
        erase_out_dir=os.path.join(opt.work_dir, 'tmp', 'erase'),
        inter_out_dir=os.path.join(opt.work_dir, 'tmp', 'inter'),
        valid_out_dir=os.path.join(opt.work_dir, 'tmp', 'valid'),
        image_path_list=image_path_list)

    if len(inter_path_list) == 0:
        return 0

    # 获取非nodata区域栅格mask
    source_mask_paths = []
    for img_path in new_img_path_list:
        shp_path = shp_temp_list[image_path_list.index(img_path)]
        preffix, _ = os.path.splitext(os.path.basename(shp_path))
        name = 'mask_' + preffix + '.tif'
        make_file(opt.work_dir)
        output_mask_path = os.path.join(opt.work_dir, 'tmp', name)
        shp2tif(
            shp_path=shp_path,
            ref_tif_path=img_path,
            target_tif_path=output_mask_path,
        )
        source_mask_paths.append(output_mask_path)

    # 生成结合表
    union_shp_list = inter_path_list[:]
    union_shp_list.append(shp_valid_bg)
    union_img_list = new_img_path_list[:]
    union_img_list.append(img_bg)

    union_shp(
        shp_path_list=union_shp_list,
        img_path_list=union_img_list,
        out_dir=os.path.join(opt.work_dir, 'tmp', 'union'))

    # 裁剪公共区域
    crop_path_list = crop_img(
        new_img_path_list,
        inter_path_list,
        output_dir=os.path.join(opt.work_dir, 'tmp', 'crop'), )

    # 获取相交区域栅格mask
    assert len(crop_path_list) == len(inter_path_list)
    mask_path_list = []
    for crop_path, inter_path in zip(crop_path_list, inter_path_list):
        name = 'mask_' + os.path.basename(inter_path)[:1] + '.tif'
        output_mask_path = os.path.join(opt.work_dir, 'tmp', name)
        shp2tif(
            shp_path=inter_path,
            ref_tif_path=crop_path,
            target_tif_path=output_mask_path
        )
        mask_path_list.append(output_mask_path)
    # print(mask_path_list)

    outpath_list = []
    outpath = None

    for j, (image_path, mask_path, source_mask_path) in enumerate(
            zip(new_img_path_list, mask_path_list, source_mask_paths)):
        # path = os.path.join(opt.work_dir, 'result')
        path = opt.work_dir
        preffix, _ = os.path.splitext(os.path.basename(image_path))

        t = time.time()
        name = str(t) + '.tif'
        outpath = os.path.join(path, name)
        outpath_list.append(outpath)

        tmp_num = len(new_img_path_list)
        global start2, end2
        start2 = start + (j / tmp_num) * (end - start)
        end2 = start + ((j + 1) / tmp_num) * (end - start)

        if j == 0:
            run_one_blend(bg_path=img_bg,
                          source_path=image_path,
                          mask_path=mask_path,
                          out_path=outpath,
                          source_mask_path=source_mask_path,
                          opt=opt
                          )
        if j != 0:
            img_bg = outpath_list[j - 1]
            run_one_blend(bg_path=img_bg,
                          source_path=image_path,
                          mask_path=mask_path,
                          out_path=outpath,
                          source_mask_path=source_mask_path,
                          opt=opt
                          )
            os.remove(outpath_list[j - 1])
    return outpath_list[-1]


def map_blend(bg_patch, s_patch, m_patch, sm_patch=None):
    bg = einsum('ijk->jki', bg_patch)
    s = einsum('ijk->jki', s_patch)
    if len(m_patch.shape) == 3:
        weight_map = einsum('ijk->jki', m_patch)
    else:
        weight_map = np.stack([m_patch, m_patch, m_patch], axis=-1)

    # tmp = np.where(s == 0)
    # if len(tmp[0]) != 0:
    #     weight_map[tmp] = 0
    if sm_patch is None:
        sm_patch = np.ones_like(weight_map)
    else:
        sm_patch = np.stack([sm_patch, sm_patch, sm_patch], axis=-1)

    try:
        weight_map[sm_patch == 0] = 0
    except:
        print(weight_map.shape, sm_patch.shape)

    weight_map = weight_map.astype(np.float64)
    s = s.astype(np.float64)
    bg = bg.astype(np.float64)

    out = ne.evaluate('bg * (1. - weight_map) + s * weight_map')
    # bg的nodata不输出
    out[bg == 0.] = 0.
    # out = out * np.array(bg, dtype=bool)
    out = einsum('ijk->kij', out)
    return out


def generate_map(m_patch):
    weight_map = m_patch
    # 膨胀
    kernel_size = 100
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    weight_map = cv2.dilate(weight_map, kernel, iterations=1)

    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size * 1.0)
    weight_map = cv2.filter2D(weight_map, -1, kernel, borderType=cv2.BORDER_CONSTANT) / 255.
    return weight_map


# 追加source的mask
def run_one_blend(bg_path, source_path, mask_path, source_mask_path, out_path, opt):
    bg = IMAGE2()
    s = IMAGE2()
    m = IMAGE2()
    s_m = IMAGE2()
    s_m.read_img(source_mask_path)

    bg.read_img(bg_path)
    s.read_img(source_path)
    m.read_img(mask_path)
    bg.copy_image(filename=out_path)
    # bg.create_img(filename=out_path)

    extent_m = [0, 0, m.im_width, m.im_height]
    m_patch = m.get_extent(extent_m)
    weight_map = generate_map(m_patch)

    bg_stat = image_stats_path(generate_overview(img_bg, opt.work_dir, pct=5)) if opt.color_trans else None
    s_stat = image_stats_path(generate_overview(source_path, opt.work_dir, pct=5)) if opt.color_trans else None

    if m.im_width * m.im_height >= 10240 * 10240:
        extents_m = GenExtents(m.im_width, m.im_height, win_size=opt.win_size)
        for i, extent_m in enumerate(extents_m):
            x_ras, y_ras, width, height = extent_m
            x_geo, y_geo = coord_ras2geo(m.im_geotrans, [x_ras, y_ras])
            # 计算bg 的栅格位置
            x_bg, y_bg = coord_geo2ras(bg.im_geotrans, [x_geo, y_geo])
            extent_bg = [x_bg, y_bg, width, height]

            # 计算source的栅格位置
            x_s, y_s = coord_geo2ras(s.im_geotrans, [x_geo, y_geo])
            extent_s = [x_s, y_s, width, height]

            # source_mask
            x_sm, y_sm = coord_geo2ras(s_m.im_geotrans, [x_geo, y_geo])
            extent_sm = [x_sm, y_sm, width, height]

            sm_patch = s_m.get_extent(extent_sm)
            bg_patch = bg.get_extent(extent_bg)
            s_patch = s.get_extent(extent_s)
            m_patch = weight_map[y_ras:y_ras + height, x_ras:x_ras + width]
            if np.max(m_patch) == 0:
                out_patch = bg_patch
            else:
                if opt.color_trans:
                    s_patch = color_transfer_para(
                        input=s_patch,
                        img_stats=s_stat,
                        ref_stats=bg_stat,
                        clip=True,
                        preserve_paper=False
                    )

                # 输入影像块进行去云
                out_patch = map_blend(bg_patch, s_patch, m_patch, sm_patch)
            # 写出
            bg.write2copy_image(extent=extent_bg, im_data=out_patch)
            # bg.write_extent(extent=extent_bg, im_data=out_patch)

            current = start2 + (i / len(extents_m)) * (end2 - start2)
            print('{}:{:.4f}'.format(opt.message, current), flush=True)
    else:
        x_ras, y_ras, width, height = extent_m
        x_geo, y_geo = coord_ras2geo(m.im_geotrans, [x_ras, y_ras])
        # 计算bg 的栅格位置
        x_bg, y_bg = coord_geo2ras(bg.im_geotrans, [x_geo, y_geo])
        extent_bg = [x_bg, y_bg, width, height]
        # 计算source的栅格位置
        x_s, y_s = coord_geo2ras(s.im_geotrans, [x_geo, y_geo])
        extent_s = [x_s, y_s, width, height]
        # source_mask
        x_sm, y_sm = coord_geo2ras(s_m.im_geotrans, [x_geo, y_geo])
        extent_sm = [x_sm, y_sm, width, height]

        sm_patch = s_m.get_extent(extent_sm)
        bg_patch = bg.get_extent(extent_bg)
        s_patch = s.get_extent(extent_s)
        m_patch = weight_map

        if opt.color_trans:
            s_patch = color_transfer_para(
                input=s_patch,
                img_stats=s_stat,
                ref_stats=bg_stat,
                clip=True,
                preserve_paper=False
            )

        out_patch = map_blend(bg_patch, s_patch, m_patch, sm_patch)
        # 写出
        bg.write2copy_image(extent=extent_bg, im_data=out_patch)
        # bg.write_extent(extent=extent_bg, im_data=out_patch)

        current = end2
        print('{}:{}'.format(opt.message, current), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('cloud removal for remote sensing images')
    parser.add_argument('--json_path', type=str, default=r'parameters.json')
    parser.add_argument('--input_image_path_list', type=list, default=None)
    parser.add_argument('--input_valid_list', type=list, default=None)
    parser.add_argument('--input_cloud_list', type=list, default=None)
    parser.add_argument('--input_temp_list', type=list, default=None)
    parser.add_argument('--work_dir', type=str, default=r'C:\tmp')
    parser.add_argument('--message', type=str, default=r'message')
    parser.add_argument('--color_trans', type=int, default=0)
    parser.add_argument('--win_size', type=int, default=2048)
    opt = parser.parse_args()
    if os.path.exists(opt.json_path):
        with open(opt.json_path, 'r', encoding='utf-8-sig') as f:
            args = json.load(f)
            opt.input_image_path_list = args['input_image_path_list']
            opt.input_valid_list = args['input_valid_list']
            opt.input_cloud_list = args['input_cloud_list']
            opt.input_temp_list = args['input_temp_list']
            opt.work_dir = args['work_dir']
            opt.message = args['message']
            opt.color_trans = int(args['color_trans'])
            opt.win_size = int(args['win_size'])

    print('--------options----------', flush=True)
    for k in list(vars(opt).keys()):
        print('%s: %s' % (k, vars(opt)[k]), flush=True)
    print('--------options----------\n', flush=True)

    t0 = time.time()
    if len(opt.input_image_path_list) == 0 or len(opt.input_valid_list) == 0:
        raise KeyError('请输入影像或shp文件')
    if len(opt.input_image_path_list) != len(opt.input_valid_list):
        raise KeyError('影像文件与矢量文件数目不一致')

    num = len(opt.input_image_path_list)
    make_file(opt.work_dir)
    # 去云
    cloud_removal_path_list = []
    for i in range(num):
        start = i / num
        end = (i + 1) / num
        valid_list = opt.input_valid_list[:]
        valid_bg = valid_list.pop(i)
        cloud_list = opt.input_cloud_list[:]
        cloud_bg = cloud_list.pop(i)
        image_path_list = opt.input_image_path_list[:]
        img_bg = image_path_list.pop(i)

        temp_list = opt.input_temp_list[:]
        temp_list.pop(i)

        # print('start cloud removal mission {}/{}'.format(i + 1, num))
        # print('bg_img: {}'.format(img_bg))
        # print('valid_bg: {}'.format(valid_bg))
        # print('cloud_bg: {}'.format(cloud_bg))
        #
        # print('image_list: {}'.format(image_path_list))
        # print('valid_list: {}\n'.format(valid_list))

        img_cloud_removal = cloud_removal(
            shp_bg=cloud_bg,
            img_bg=img_bg,
            shp_valid_bg=valid_bg,
            image_path_list=image_path_list,
            shp_path_list=valid_list,
            shp_temp_list=temp_list,
            opt=opt
        )
        if img_cloud_removal != 0:
            cloud_removal_path_list.append(img_cloud_removal)
    #
    # # # 镶嵌
    # # t = time.time()
    # # raster_mosaic(
    # #     file_path_list=cloud_removal_path_list[::-1],
    # #     output_path=os.path.join(work_dir, r'result\mosaic_{}.tif'.format(t))
    # # )
    # print('{}:{}'.format(opt.message, 1.), flush=True)
    # print("CostTime:{}".format(time.time() - t0), flush=True)
