import numpy as np


def histMatch(img, ref):
    out = np.zeros_like(img)
    _, _, colorChannel = img.shape
    for i in range(colorChannel):
        # get the histogram
        hist_img, _ = np.histogram(img[:, :, i], 256, range=(0, 255))
        hist_ref, _ = np.histogram(ref[:, :, i], 256, range=(0, 255))
        # get the accumulative histogram
        cdf_img = np.cumsum(hist_img)
        cdf_ref = np.cumsum(hist_ref)
        # match
        for j in range(1, 256):
            tmp = abs(cdf_img[j] - cdf_ref)
            tmp = tmp.tolist()
            # find the smallest number in tmp, get the index of this number
            idx = tmp.index(min(tmp))
            out[:, :, i][img[:, :, i] == j] = idx
    return out