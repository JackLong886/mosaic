

from osgeo import gdal, ogr, osr
import numpy as np
import os
from glob import glob
from math import ceil
import time


def GetExtent(infile):
    ds = gdal.Open(infile)
    geotrans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x, max_y = geotrans[0], geotrans[3]
    max_x, min_y = geotrans[0] + xsize * geotrans[1], geotrans[3] + ysize * geotrans[5]
    ds = None
    return min_x, max_y, max_x, min_y


def RasterMosaic(file_list, outpath):
    Open = gdal.Open
    min_x, max_y, max_x, min_y = GetExtent(file_list[0])
    for infile in file_list:
        minx, maxy, maxx, miny = GetExtent(infile)
        min_x, min_y = min(min_x, minx), min(min_y, miny)
        max_x, max_y = max(max_x, maxx), max(max_y, maxy)

    in_ds = Open(file_list[0])
    in_band = in_ds.GetRasterBand(1)
    geotrans = list(in_ds.GetGeoTransform())
    width, height = geotrans[1], geotrans[5]
    columns = ceil((max_x - min_x) / width)  # 列数
    rows = ceil((max_y - min_y) / (-height))  # 行数

    outfile = outpath + file_list[0][:4] + '.tif'  # 结果文件名，可自行修改
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(outfile, columns, rows, 1, in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] = min_x  # 更正左上角坐标
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)
    out_band = out_ds.GetRasterBand(1)
    inv_geotrans = gdal.InvGeoTransform(geotrans)

    for in_fn in file_list:
        in_ds = Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)

        data = in_ds.GetRasterBand(1).ReadAsArray()
        out_band.WriteArray(data, x, y)  # x，y是开始写入时左上角像元行列号
    del in_ds, out_band, out_ds
    return outfile


def compress(path, target_path, method="LZW"):  #
    """使用gdal进行文件压缩，
          LZW方法属于无损压缩，
          效果非常给力，4G大小的数据压缩后只有三十多M"""
    dataset = gdal.Open(path)
    driver = gdal.GetDriverByName('GTiff')
    driver.CreateCopy(target_path, dataset, strict=1, options=["TILED=YES", "COMPRESS={0}".format(method)])
    del dataset


if __name__ == '__main__':
    path = r'J:\backup'  # 该文件夹下存放了待拼接的栅格
    os.chdir(path)
    raster_list = sorted(glob('*.tif'))  # 读取文件夹下所有tif数据
    result = RasterMosaic(raster_list, outpath=r'J:\backup\Global')  # 拼接栅格
    compress(result, target_path=r'J:\backup\Global.tif')  # 压缩栅格

def ras2shp(folder):
    os.chdir(folder)  # 设置默认路径
    for raster in os.listdir():  # 遍历路径中每一个文件，如果存在gdal不能打开的文件类型，则后续代码可能会报错。
        inraster = gdal.Open(raster)  # 读取路径中的栅格数据
        inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段，如果是单波段数据的话那就都是1
        prj = osr.SpatialReference()
        prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息，用来为后面生成的矢量做准备

        outshp = raster[:-4] + ".shp"  # 给后面生成的矢量准备一个输出文件名，这里就是把原栅格的文件名后缀名改成shp了
        drv = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outshp):  # 若文件已经存在，则删除它继续重新做一遍
            drv.DeleteDataSource(outshp)
        Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
        Poly_layer = Polygon.CreateLayer(raster[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)  # 对shp文件创建一个图层，定义为多个面类
        newField = ogr.FieldDefn('value', ogr.OFTReal)  # 给目标shp文件添加一个字段，用来存储原始栅格的pixel value
        Poly_layer.CreateField(newField)

        gdal.FPolygonize(inband, None, Poly_layer, 0)  # 核心函数，执行的就是栅格转矢量操作
        Polygon.SyncToDisk()
        Polygon = None