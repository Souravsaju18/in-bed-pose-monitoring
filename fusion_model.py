import numpy as np

def fuse_features(depth_img, thermal_img):
    return (depth_img.astype("float") + thermal_img.astype("float")) / 2