from osgeo import gdal
import os
import numpy as np
from data.MyDatasets import make_dataset_2


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


class IMAGE2:
    # 读图像文件
    def read_img(self, filename):
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

    def create_img(self, filename, out_bands, datatype=gdal.GDT_UInt16):
        self.datatype = datatype
        self.out_bands = out_bands
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")
        self.output_dataset = driver.Create(filename, self.im_width, self.im_height, out_bands, datatype)
        self.output_dataset.SetGeoTransform(self.im_geotrans)  # 写入仿射变换参数
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


def gdal_cut(file_path, cut_size=256, win_std=0):
    image = IMAGE2()
    image.read_img(file_path)
    extents = GenExtents(image.im_width, image.im_height, cut_size, win_std)
    print(image.im_width, image.im_height)
    file = os.path.dirname(file_path)
    prefix = os.path.splitext(os.path.basename(file_path))[0]
    out_file = os.path.join(file + prefix)
    if not os.path.exists(out_file):
        os.makedirs(out_file)
    for extent in extents:
        img = image.get_extent(extent)
        x, y, _, _ = extent
        out_name = '{0}_{1}_{2}.png'.format(x, y, prefix)
        out_path = os.path.join(out_file, out_name)
        writeTiff(img, im_geotrans=image.im_geotrans,
                  im_proj=image.im_proj, path=out_path)


#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans)
        dataset.SetProjection(im_proj)
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


if __name__ == '__main__':
    file_path = r'C:\LCJ\image_data\GF6\resize'
    images = make_dataset_2(file_path)
    i = 0
    for image in images:
        gdal_cut(image)
