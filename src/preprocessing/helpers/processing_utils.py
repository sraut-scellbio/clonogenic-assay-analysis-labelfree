import os
import random
import re
import cv2
import pdb
import glob
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from skimage.filters import threshold_otsu
import tifffile

def disp_img(*imgs, title=''):
    if len(imgs) == 1:
        # Display one image
        plt.imshow(imgs[0], cmap='gray')
        plt.axis('off')
        plt.title(title)
        plt.show()
    elif len(imgs) == 2:
        # Display two images side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, img in zip(axes, imgs):
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.title(title)
        plt.show()
    else:
        raise ValueError("This function only supports displaying one or two images.")


def is_edge_frame(filename, grid_rows, grid_cols):
    pattern = r"_(\d+)_(\d+)\.tif$"

    match = re.search(pattern, filename)
    if match:
        row_val = int(match.group(1))  # Extract row number
        col_val = int(match.group(2))  # Extract column number
        if (1 < row_val < grid_rows-2) and (1 < col_val < grid_cols-2) :
            return False
        else:
            return True
    else:
        print("No match found.")
        return False

def get_normalized_8bit_stack(img_arr):
    min_vals = img_arr.min(axis=(1, 2))[:, np.newaxis, np.newaxis]
    max_vals = img_arr.max(axis=(1, 2))[:, np.newaxis, np.newaxis]
    norm_arr_scaled = ((img_arr - min_vals) / (max_vals - min_vals)) * 255
    img_stack_8bit = norm_arr_scaled.astype(np.uint8)
    return img_stack_8bit

def get_background_profile(
    image_dir,
    channel_name,
    n_samples=50,
    pct=20,
    blur_sigma=15,
    rfp_open_size=201
):
    """
    Returns:
      thresh               = Otsu (or high-percentile for RFP) on pooled 8-bit
      background_profile   = 2D float32 array to subtract per-frame
      norm_min, norm_max   = scalars for global min/max mapping to 0–255
    """
    # 1) gather & sample files
    patterns = ['*.jpg','*.jpeg','*.png','*.tif','*.tiff']
    files = sum([glob.glob(os.path.join(image_dir, p)) for p in patterns], [])
    if not files:
        raise ValueError(f"No image files in {image_dir}")
    sample = random.sample(files, min(n_samples, len(files)))

    # 2) read & stack raw (uint16) frames → shape = (N, H, W)
    stacks = []
    for fp in sample:
        arr = tifffile.imread(fp)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        stacks.append(arr)
    raw_stacks = np.concatenate(stacks, axis=0).astype(np.float32)

    # 3) percentile‑based, per‑pixel background
    background_profile = np.percentile(raw_stacks, pct, axis=0).astype(np.float32)

    # 4) **RFP‑specific smoothing**: remove any small bumps, keep only the smooth gradient
    if channel_name.lower() == "rfp":
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (rfp_open_size, rfp_open_size)
        )
        # apply opening directly to the profile
        # (uint8 conversion is OK here since it's just for smoothing)
        bp_uint8 = np.clip(background_profile, 0, 255).astype(np.uint8)
        background_profile = cv2.morphologyEx(bp_uint8, cv2.MORPH_OPEN, kern).astype(np.float32)

    # 5) final Gaussian blur to remove any leftover high‑freq noise
    background_profile = cv2.GaussianBlur(
        background_profile, (0,0), sigmaX=blur_sigma
    )

    return background_profile


def is_image_blurred(image_path, label=False, laplacian_threshold=10, edge_ratio_threshold=0.000001):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'")

    # Step 1: Calculate the Laplacian variance
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()

    # Step 2: Perform Canny edge detection
    edges = cv2.Canny(image, 100, 200)
    edge_pixel_count = np.sum(edges > 0)
    total_pixel_count = image.size
    edge_ratio = edge_pixel_count / total_pixel_count

    # Debugging output
    # if edge_ratio < edge_ratio_threshold:
    #     disp_img(image)
    #     print(f"Edge Pixel Ratio: {edge_ratio}")
    #     print(f"Is Edge Ratio < Threshold? {edge_ratio < edge_ratio_threshold}")


    # Step 3: Combine both checks to determine if the image is blurred
    if edge_ratio < edge_ratio_threshold:
        return True  # Image is blurred
    else:
        return False  # Image is not blurred