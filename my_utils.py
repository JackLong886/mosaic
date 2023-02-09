import os
from osgeo import gdal, ogr, osr
import numpy as np
import cv2

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


def make_file(path):
    if not os.path.exists(path):
        os.makedirs(path)


def coord_ras2geo(im_geotrans, coord):
    x0, x_res, _, y0, _, y_res = im_geotrans
    x = x0 + x_res * coord[0]
    y = y0 + y_res * coord[1]
    return x, y


def coord_geo2ras(im_geotrans, coord):
    x0, x_res, _, y0, _, y_res = im_geotrans
    x = int(round(abs((coord[0] - x0)) / abs(x_res)))
    y = int(round(abs((coord[1] - y0)) / abs(y_res)))
    return x, y


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


def generate_map(m_patch):
    weight_map = m_patch
    kernel_size = 100
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    weight_map = cv2.dilate(weight_map, kernel, iterations=1)

    kernel_size = 100
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size * 1.0)
    weight_map = cv2.filter2D(weight_map, -1, kernel, borderType=cv2.BORDER_REPLICATE) / 255.
    return weight_map


def generate_overview(img_path, work_dir, pct=5):
    basename = os.path.basename(img_path)
    preffix, _ = os.path.splitext(basename)
    out_path = os.path.join(work_dir, preffix + '_overview.tif')
    options = gdal.TranslateOptions(
        format='GTiff',
        # width=500, height=500
        heightPct=pct, widthPct=pct
    )
    gdal.Translate(
        destName=out_path,
        srcDS=img_path,
        options=options
    )
    return out_path

# 擦除
def erase_cloud_mask(to_erase, erase_list, erase_out_dir, inter_out_dir, vaild_out_dir, image_path_list):
    make_file(erase_out_dir)
    make_file(inter_out_dir)
    # make_file(vaild_out_dir)
    # 开始批量擦除
    erase_path_list = []
    inter_path_list = []
    vaild_path_list = []
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


def run_one_erase(to_erase, erase, erase_out_dir, inter_out_dir):
    make_file(erase_out_dir)
    make_file(inter_out_dir)
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # 被擦除shp
    to_erase_shp = driver.Open(to_erase)
    to_erase_layer = to_erase_shp.GetLayer()
    num_feature = to_erase_layer.GetFeatureCount()
    to_erase_srs = to_erase_layer.GetSpatialRef()
    to_erase_defn = to_erase_layer.GetLayerDefn()

    if num_feature == 0:
        return 0

    basename = os.path.basename(erase)
    dst_erase = os.path.join(erase_out_dir, 'erase_' + basename)
    dst_inter = os.path.join(inter_out_dir, 'inter_' + basename)

    outds_inter = driver.CreateDataSource(dst_inter)
    outlayer_inter = outds_inter.CreateLayer(dst_inter, srs=to_erase_srs, geom_type=ogr.wkbPolygon)

    outds_erase = driver.CreateDataSource(dst_erase)
    outlayer_erase = outds_erase.CreateLayer(dst_erase, srs=to_erase_srs, geom_type=ogr.wkbPolygon)

    for j in range(to_erase_defn.GetFieldCount()):
        outlayer_inter.CreateField(to_erase_defn.GetFieldDefn(j))
        outlayer_erase.CreateField(to_erase_defn.GetFieldDefn(j))

    to_erase_layer.Erase(to_erase_layer, outlayer_erase)
    to_erase_layer.Intersection(to_erase_layer, outlayer_inter)
    to_erase_shp.Destroy()

    return dst_erase, dst_inter


