import json
import os.path

input_image_path_list = [
    r'C:\Users\DELL\Desktop\l123\l1_8bit.tif',
    r'C:\Users\DELL\Desktop\l123\l2_8bit.tif',
    r'C:\Users\DELL\Desktop\l123\l3_8bit.tif',
]

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

work_dir = r'C:\Users\DELL\Desktop\CloudRemoval\test7'

para = {
    "input_image_path_list": input_image_path_list,
    "input_valid_list": input_valid_list,
    "input_cloud_list": input_cloud_list,
    "input_temp_list": input_temp_list,
    "work_dir": work_dir,
    "win_size": 1024,
    "color_trans": 1,
    "message": 'message',
}

with open('parameters.json', 'w') as f:
    json.dump(para, f)