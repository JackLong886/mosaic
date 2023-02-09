import cv2

img1 = cv2.imread(r'C:\Users\DELL\Desktop\CloudRemoval\test\ceshi_result\tmp.tif')
img2 = cv2.imread(r'C:\Users\DELL\Desktop\CloudRemoval\test\ceshi_result\tmp_bg.tif')
mask = cv2.imread(r'C:\Users\DELL\Desktop\CloudRemoval\test\ceshi_result\tmp_mask.tif')
outpath = r'C:\Users\DELL\Desktop\CloudRemoval\test\ceshi_result\tmp2.tif'
mask[mask != 255] = 0
tmp1 = cv2.bitwise_and(img1, mask)
mask_inv = cv2.bitwise_not(mask)
tmp2 = cv2.bitwise_and(img2, mask_inv)
out = cv2.add(tmp1, tmp2)

cv2.imwrite(outpath, out)
print('ok')
# cv2.imshow('1', out)
# cv2.waitKey(0)