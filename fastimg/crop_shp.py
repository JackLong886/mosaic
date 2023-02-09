import os.path

from osgeo import ogr, gdal
from cloud_removal0209 import make_file
input_valid_list = [
    r'C:\Users\DELL\Desktop\CloudRemoval\CloudDetect\out\valid1.shp',
    r'C:\Users\DELL\Desktop\CloudRemoval\CloudDetect\out\valid2.shp',
    r'C:\Users\DELL\Desktop\CloudRemoval\CloudDetect\out\valid3.shp',
]

input_cloud_list = [
    r'C:\Users\DELL\Desktop\CloudRemoval\CloudDetect\out\cloud1.shp',
    r'C:\Users\DELL\Desktop\CloudRemoval\CloudDetect\out\cloud2.shp',
    r'C:\Users\DELL\Desktop\CloudRemoval\CloudDetect\out\cloud3.shp',
]

input_temp_list = [
    r'C:\Users\DELL\Desktop\CloudRemoval\CloudDetect\out\temp1.shp',
    r'C:\Users\DELL\Desktop\CloudRemoval\CloudDetect\out\temp2.shp',
    r'C:\Users\DELL\Desktop\CloudRemoval\CloudDetect\out\temp3.shp',
]

work_dir = r'C:\Users\DELL\Desktop\CloudRemoval\CloudDetect\out2'
make_file(work_dir)
driver = ogr.GetDriverByName('ESRI Shapefile')
for valid, cloud, temp in zip(input_valid_list, input_cloud_list, input_temp_list):
    ds_temp = driver.Open(temp)
    layer_temp = ds_temp.GetLayerByIndex(0)

    ds_valid = driver.Open(valid, 1)
    layer_valid = ds_valid.GetLayerByIndex(0)
    srs_valid = layer_valid.GetSpatialRef()
    defn_valid = layer_valid.GetLayerDefn()
    dst_valid = os.path.join(work_dir, os.path.basename(valid))
    outds_valid = driver.CreateDataSource(dst_valid)
    outlayer_valid = outds_valid.CreateLayer(dst_valid, srs=srs_valid, geom_type=ogr.wkbPolygon)
    for i in range(defn_valid.GetFieldCount()):
        outlayer_valid.CreateField(defn_valid.GetFieldDefn(i))
    layer_valid.Intersection(layer_temp, outlayer_valid)

    ds_cloud = driver.Open(cloud, 1)
    layer_cloud = ds_cloud.GetLayerByIndex(0)
    srs_cloud = layer_cloud.GetSpatialRef()
    defn_cloud = layer_cloud.GetLayerDefn()
    dst_cloud = os.path.join(work_dir, os.path.basename(cloud))
    outds_cloud = driver.CreateDataSource(dst_cloud)
    outlayer_cloud = outds_cloud.CreateLayer(dst_cloud, srs=srs_cloud, geom_type=ogr.wkbPolygon)
    for i in range(defn_cloud.GetFieldCount()):
        outlayer_cloud.CreateField(defn_cloud.GetFieldDefn(i))
    layer_cloud.Intersection(layer_temp, outlayer_cloud)

    ds_temp.Destroy()
    ds_cloud.Destroy()
    ds_valid.Destroy()