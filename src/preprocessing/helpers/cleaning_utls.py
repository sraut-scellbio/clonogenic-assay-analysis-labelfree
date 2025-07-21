import cv2
import numpy as np
from .processing_utils import disp_img

def is_almost_black(image_path, th1=10, th2=0.90, label=False, debug=False):

    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    black_pixels = np.sum(gray_image < th1)
    total_pixels = gray_image.size
    black_ratio = black_pixels / total_pixels
    empty_frame = black_ratio >= th2

    if debug:
        disp_img(gray_image)
        if empty_frame:
            print(f"Frame empty. Black_ratio:{black_ratio}.")
        else:
            print(f"Non-empty frame. Black_ratio:{black_ratio}.")
    return empty_frame

def is_edge_frame(image_path, th1=10, th2=0.03, th3=0.90, label=False, debug=False):

    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    black_pixels = np.sum(gray_image < th1)
    total_pixels = gray_image.size
    black_ratio = black_pixels / total_pixels
    edge_frame = th2 < black_ratio < th3

    if debug:
        disp_img(gray_image)
        if edge_frame:
            print(f"Edge frame. Black ratio:{black_ratio}.")
        else:
            print(f"Non-edge frame. Black_ratio:{black_ratio}.")
    return edge_frame