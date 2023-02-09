import cv2
import numpy as np
import torch
from torchvision import transforms
from pietorch import blend
import time
t0 = time.time()

target_path = r'C:\Users\DELL\Desktop\CloudRemoval\CloudData\N23.0_trans.tif'
mask_path = r'C:\Users\DELL\Desktop\CloudRemoval\CloudData\mask23.0_trans.tif'
source_path = r'C:\Users\DELL\Desktop\CloudRemoval\CloudData\N22.9_trans.tif'
output_path = r'C:\Users\DELL\Desktop\CloudRemoval\CloudData/result23.0_torch.tif'
target = cv2.imread(target_path)
source = cv2.imread(source_path)
mask = cv2.imread(mask_path)

place = np.where(mask[:, :, 0] > 0)
x_start, y_start = np.min(place[0]), np.min(place[1])
x_end, y_end = np.max(place[0]), np.max(place[1])
print(x_start, y_start, x_end, y_end)

trans = transforms.ToTensor()
target = trans(target)
source = trans(source)
mask = trans(mask)[0, :, :]
mask[mask != 0] = 1
left_up_coord = torch.tensor([int(y_start), int(x_start)])

result = blend(target, source, mask, left_up_coord, False, channels_dim=0)
result *= 255
result = torch.clip(result, 0, 255)
result = torch.movedim(result, 0, -1).type(torch.ByteTensor).numpy()
cv2.imwrite(output_path, result)
print(time.time() - t0)
