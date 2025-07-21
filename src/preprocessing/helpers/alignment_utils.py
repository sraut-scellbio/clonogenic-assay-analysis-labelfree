import os
import cv2
import pdb
import re
import natsort
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from .processing_utils import disp_img
from .re_patterns import get_file_pattern

def find_min_max_row_col(filenames):
    pattern = r"_(\d+)_(\d+)\.tif$"

    row_values = []
    col_values = []

    for filename in filenames:
        match = re.search(pattern, filename)
        if match:
            row, col = map(int, match.groups())
            row_values.append(row)
            col_values.append(col)

    if row_values and col_values:
        row_values.sort()
        col_values.sort()

        # Finding min, max, and middle values
        min_row = min(row_values)
        max_row = max(row_values)
        mid_row = row_values[len(row_values) // 2]

        min_col = min(col_values)
        max_col = max(col_values)
        mid_col = col_values[len(col_values) // 2]

        # Compute number of rows and columns
        num_rows = max_row - min_row + 1
        num_cols = max_col - min_col + 1

        return min_row, max_row, mid_row, min_col, max_col, mid_col, num_rows, num_cols

    else:
        return None

# def get_edge_frames(well_fpath, params, well_ID, device_ID, n_grid_rows, n_grid_cols):
#
#     pattern = r"^(.+?)_(\d+)_(\d+)\.tif$"
#
#     day1_fpath = os.path.join(well_fpath, 'day1', params["channels"][0])
#     dayn_fpath = os.path.join(well_fpath, 'dayn', params["channels"][0])
#
#     fnames = os.listdir(day1_fpath)
#
#     # find min/max row/col vals in cleaned data
#     min_row, max_row, mid_row, min_col, max_col, mid_col, num_tr_rows, num_tr_cols = find_min_max_row_col(fnames)
#
#     # create edge frame names
#     edge_fnames_dict = {
#         'top': f"{device_ID}_{well_ID}_{min_row}_{mid_col}.tif",
#         'bottom': f"{device_ID}_{well_ID}_{max_row}_{mid_col}.tif",
#         'left': f"{device_ID}_{well_ID}_{mid_row}_{min_col}.tif",
#         'right': f"{device_ID}_{well_ID}_{mid_row}_{max_col}.tif"
#     }
#
#     # pdb.set_trace()
#
#     # read edge frames from day 1 and day n
#     def read_image(file_path):
#         """Reads a grayscale image and converts it to a NumPy array (uint8)."""
#         if os.path.exists(file_path):
#             img = Image.open(file_path) # Convert to grayscale
#             return np.array(img)
#         return None  # Return None if the file is not found
#
#     edge_images = {
#         'day1': {},
#         'dayn': {}
#     }
#
#     # Read edge frames from day1 and dayn
#     for edge, fname in edge_fnames_dict.items():
#         day1_img_path = os.path.join(day1_fpath, fname)
#         dayn_img_path = os.path.join(dayn_fpath, fname)
#
#         # print(f"reading {edge} edge images for {well_ID}\n")
#         d1_edge_img = read_image(day1_img_path)
#         dn_edge_img = read_image(dayn_img_path)
#         edge_images['day1'][edge] = d1_edge_img
#         edge_images['dayn'][edge] = dn_edge_img
#         # disp_img(d1_edge_img, dn_edge_img)
#
#     return edge_images, num_tr_rows, num_tr_cols, min_row, min_col


def get_montage_from_frames(img_stack_path,
                                   daystr,
                                   channel,
                                   grid_sz,
                                   params,
                                   res_fld_path,
                                   save=False):

    well_ID = os.path.basename(os.path.dirname(os.path.dirname(img_stack_path)))
    n_c, n_r = tuple(params["frame_shape"])  # Frame size (columns, rows)
    grid_rows = grid_sz[0]
    grid_cols = grid_sz[1]
    total_frames = grid_cols * grid_rows

    # Create an empty montage image
    montage_image = np.zeros((n_r * grid_rows, n_c * grid_cols), dtype=np.uint16)
    pattern = get_file_pattern("cytation", "orig")

    # check if the folder has the correct number of frames
    if len(os.listdir(img_stack_path)) != total_frames:
        print(f"the number of frames in folder {img_stack_path} does not match the size of the grid in the metafile.")

    for fname in os.listdir(img_stack_path):

        file_path = os.path.join(img_stack_path, fname)
        try:
            # Open the file as an image
            with Image.open(file_path) as img:
                img_array = np.array(img)

                # Compute row and column index
                match = pattern.match(fname)
                if not match:
                    print(f"Filename did not match pattern: {file_path.name}")
                    continue

                groups = match.groupdict()
                frame_id = int(groups.get("frame_id"))
                frame_index = frame_id - 1  # Convert to 0-based index
                row_idx = frame_index // grid_cols
                col_idx = frame_index % grid_cols

                # Compute pixel positions in the montage image
                start_y = row_idx * n_r
                start_x = col_idx * n_c

                # Place the frame in the correct position
                montage_image[start_y:start_y + n_r, start_x:start_x + n_c] = img_array

        except IOError:
            print(f"Skipping non-image file: {file_path}")

    # Save the final montage image
    if save:
        device_ID = params["device_ID"]
        montage_img_name = f"{device_ID}_{well_ID}_{daystr}_{channel}.tif"
        res_fld_path = os.path.join(res_fld_path, device_ID, well_ID, daystr, channel)
        os.makedirs(res_fld_path, exist_ok=True)
        res_fname = os.path.join(res_fld_path, montage_img_name)
        montage_img_pil = Image.fromarray(montage_image)
        montage_img_pil.save(res_fname)
        print(f"Montage image saved to {res_fname}")
    else:
        return montage_image

def calculate_shift_montage(montage_img1, montage_img2, debug=False):
    def preprocess_and_find_rect(image):
        """Thresholds the image, finds contours, and fits a bounding rectangle to the largest contour."""
        if image is None:
            return None  # Return None if the image is missing

        image_height, image_width = image.shape  # Get image dimensions

        if image.dtype == np.uint16:
            min_val = image.min()
            max_val = image.max()
            if max_val > min_val:  # Prevent division by zero
                image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

        # Apply threshold to create a binary mask
        _, binary_mask = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None  # No contours found

        # Find bounding rectangle for the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Compute area of the bounding rectangle and image
        rect_area = w * h
        image_area = image_width * image_height

        # If the detected object covers more than 95% of the image, return None
        if rect_area / image_area > 0.99:
            print(f"Rectangle area greater than image area! Ratio = {rect_area / image_area}")
            return None

        if debug:
            # Draw the bounding rectangle on the image (Red color)
            image_with_rect = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(image_with_rect, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display the image with the bounding box
            plt.figure(figsize=(5, 5))
            plt.imshow(image_with_rect, cmap="gray")
            plt.axis("off")
            plt.show()

        return x, y, w, h  # Return bounding box (x, y, width, height)

    # Get bounding rectangles for both montage images
    rect1 = preprocess_and_find_rect(montage_img1)
    rect2 = preprocess_and_find_rect(montage_img2)

    # If bounding boxes cannot be determined, return None
    if rect1 is None or rect2 is None:
        print("Bounding rectangle could not be determined for one or both montage images.")
        return {"top": None, "bottom": None, "left": None, "right": None}

    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    # Compute shifts
    shifts = {
        "top": y2 - y1,  # Shift in top edge
        "bottom": (y2 + h2) - (y1 + h1),  # Shift in bottom edge
        "left": x2 - x1,  # Shift in left edge
        "right": (x2 + w2) - (x1 + w1),  # Shift in right edge
    }

    # Handle cases where one shift value is None
    if shifts["top"] is None and shifts["bottom"] is not None:
        shifts["top"] = shifts["bottom"]
    elif shifts["bottom"] is None and shifts["top"] is not None:
        shifts["bottom"] = shifts["top"]
    elif shifts["top"] is None and shifts["bottom"] is None:
        shifts["top"] = shifts["bottom"] = 0  # Set both to 0 if both are None

    if shifts["left"] is None and shifts["right"] is not None:
        shifts["left"] = shifts["right"]
    elif shifts["right"] is None and shifts["left"] is not None:
        shifts["right"] = shifts["left"]
    elif shifts["left"] is None and shifts["right"] is None:
        shifts["left"] = shifts["right"] = 0  # Set both to 0 if both are None

    return shifts


def align_and_crop_frames(shift_dict,
                      well_fpath,
                      start_sname,
                      dst_well_fpath,
                      day_fldname,
                      num_tr_grid_rows,
                      num_tr_grid_cols,
                      row_start,
                      col_start,
                      overlap_x,
                      overlap_y,
                      microns_per_pixel,
                      debug=False):

    # Automatically detect channels inside the specified day folder
    day_fldroot = well_fpath / day_fldname
    channels = [d.name for d in day_fldroot.iterdir() if d.is_dir()]

    # only crop if overlap is greater than zero
    if overlap_y > 0 or overlap_x > 0:
        # compute crop params
        crop_x = int((overlap_x / 2) / microns_per_pixel)  # divide overlap by 2 to include half of the overlap in each image
        crop_y = int((overlap_y / 2) / microns_per_pixel)
        crop_frame = True
    else:
        crop_x = None
        crop_y = None
        crop_frame = False

    for channel in channels:
        print(f"Aligning channel: {channel} for well {well_fpath.name}.")

        # Construct paths to channle folders
        start_s_ch_fpath = well_fpath / start_sname / channel
        dayn_s_ch_fpath = well_fpath / day_fldname / channel

        # Ensure output folder exists
        corrected_fpath = dst_well_fpath / day_fldname / channel
        os.makedirs(corrected_fpath, exist_ok=True)

        # Process all images in Day N
        fnames = os.listdir(start_s_ch_fpath)
        sorted_fnames = natsort.natsorted(fnames)

        # Regex pattern to extract frame_id from filenames
        pattern = get_file_pattern("cytation", "orig")

        for fname in sorted_fnames:
            match = pattern.match(fname)
            if not match:
                print(f"Filename did not match pattern: {fname}")
                continue

            img_path_n = os.path.join(dayn_s_ch_fpath, fname)
            imgn = cv2.imread(img_path_n, cv2.IMREAD_GRAYSCALE)

            if shift_dict is not None:

                # Extract shifts
                top_shift = shift_dict.get("top", 0)
                bottom_shift = shift_dict.get("bottom", 0)
                left_shift = shift_dict.get("left", 0)
                right_shift = shift_dict.get("right", 0)

                # Compute per-tile shift deltas
                delta_x = (right_shift - left_shift) / max(num_tr_grid_cols - 1, 1)
                delta_y = (bottom_shift - top_shift) / max(num_tr_grid_rows - 1, 1)

                groups = match.groupdict()
                frame_id = int(groups.get("frame_id"))

                frame_index = frame_id - 1  # Convert to 0-based index
                row_idx = frame_index // num_tr_grid_cols
                col_idx = frame_index % num_tr_grid_cols

                # Compute shift for this image
                x_shift = left_shift + (col_idx - col_start) * delta_x
                y_shift = top_shift + (row_idx - row_start) * delta_y

                # Create transformation matrix
                H = np.array([
                    [1, 0, -x_shift],
                    [0, 1, -y_shift]
                ], dtype=np.float32)

                if imgn is None:
                    print(f"Warning: Could not read {fname}. Skipping...")
                    continue

                # Apply translation
                corrected_img = cv2.warpAffine(imgn, H, (imgn.shape[1], imgn.shape[0]), flags=cv2.INTER_LINEAR)

                # crop image
                if crop_frame:
                    corrected_img = corrected_img[crop_y:-crop_y, crop_x:-crop_x]

                if debug:
                    # Read reference and target images
                    img_path_1 = os.path.join(start_s_ch_fpath, fname)
                    img1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
                    disp_img(img1, imgn)
                    disp_img(img1, corrected_img)

            # for day1 image only apply the crop
            else:

                if crop_frame:
                    # crop image according to overlap
                    corrected_img = imgn[crop_y:-crop_y, crop_x:-crop_x]
                else:
                    corrected_img = imgn

                if debug:
                    # Read reference and target images
                    img_path_1 = os.path.join(start_s_ch_fpath, fname)
                    img1 = cv2.imread(img_path_1, cv2.IMREAD_GRAYSCALE)
                    disp_img(img1, corrected_img)

            # Save corrected image
            corrected_img_path = os.path.join(corrected_fpath, fname)
            cv2.imwrite(corrected_img_path, corrected_img)


# def calculate_shift(edge_images):
#     def preprocess_and_find_rect(image):
#         """Thresholds the image, finds contours, and fits a bounding rectangle to the largest contour."""
#         if image is None:
#             return None  # Return None if the image is missing
#
#         image_height, image_width = image.shape  # Get image dimensions
#
#         if image.dtype == np.uint16:
#             min_val = image.min()
#             max_val = image.max()
#             if max_val > min_val:  # Prevent division by zero
#                 image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
#             else:
#                 image = np.zeros_like(image, dtype=np.uint8)
#
#         # Apply threshold to create a binary mask
#         _, binary_mask = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
#
#         # Find contours
#         contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         if not contours:
#             return None  # No contours found
#
#         # Find bounding rectangle for the largest contour
#         largest_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#
#         # Compute area of the bounding rectangle and image
#         rect_area = w * h
#         image_area = image_width * image_height
#
#         # If the detected object covers more than 95% of the image, return None
#         if rect_area / image_area > 0.99:
#             print(f"Rectangle area greater than image area! Ratio = {rect_area / image_area}")
#             return None
#
#         # # Draw the bounding rectangle on the image (Red color)
#         # image_with_rect = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#         # cv2.rectangle(image_with_rect, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         #
#         # # Display the image with the bounding box
#         # plt.figure(figsize=(5, 5))
#         # plt.imshow(image_with_rect, cmap="gray")
#         # plt.axis("off")
#         # plt.show()
#
#         return x, y, w, h  # Return bounding box (x, y, width, height)
#
#     shifts = {}
#
#     for edge in ['top', 'bottom', 'left', 'right']:
#         rect1 = preprocess_and_find_rect(edge_images['day1'].get(edge))
#         rectn = preprocess_and_find_rect(edge_images['dayn'].get(edge))
#
#         if rect1 is None or rectn is None:
#             print(f"Shift could not be calculated for {edge} because the detected object covers almost the entire image or is missing.")
#             shifts[edge] = None  # If either rectangle is missing or too large, set shift to None
#             continue
#
#         x1, y1, w1, h1 = rect1
#         x2, y2, w2, h2 = rectn
#
#         if edge == "top":
#             shifts[edge] = y2 - y1  # Difference in top edges (vertical shift)
#         elif edge == "bottom":
#             shifts[edge] = (y2 + h2) - (y1 + h1)  # Difference in bottom edges (vertical shift)
#         elif edge == "left":
#             shifts[edge] = x2 - x1  # Difference in left edges (horizontal shift)
#         elif edge == "right":
#             shifts[edge] = (x2 + w2) - (x1 + w1)  # Difference in right edges (horizontal shift)
#
#         # Handle cases where one shift value is None
#     if shifts["top"] is None and shifts["bottom"] is not None:
#         shifts["top"] = shifts["bottom"]
#     elif shifts["bottom"] is None and shifts["top"] is not None:
#         shifts["bottom"] = shifts["top"]
#     elif shifts["top"] is None and shifts["bottom"] is None:
#         shifts["top"] = shifts["bottom"] = 0  # Set both to 0 if both are None
#
#     if shifts["left"] is None and shifts["right"] is not None:
#         shifts["left"] = shifts["right"]
#     elif shifts["right"] is None and shifts["left"] is not None:
#         shifts["right"] = shifts["left"]
#     elif shifts["left"] is None and shifts["right"] is None:
#         shifts["left"] = shifts["right"] = 0  # Set both to 0 if both are None
#
#     return shifts




