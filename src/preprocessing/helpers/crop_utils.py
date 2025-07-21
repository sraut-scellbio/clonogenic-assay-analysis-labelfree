import pdb
import sys

import cv2
import os

import psutil
import tifffile
import uuid
import math

from tifffile import TiffFile
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

parent_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(str(parent_dir))

from src.preprocessing.coeffs.film_params import well_width_microns, bridge_width_microns
from src.processing.well_identification import interpolate_missing_points

def get_crop_box_within_bounds(center_x, center_y, b_x, b_y, b_w, b_h):
    """
    Compute the largest possible crop centered at (center_x, center_y)
    that stays within the bounds defined by (b_x, b_y, b_w, b_h).
    """
    bx1, by1 = b_x, b_y
    bx2, by2 = b_x + b_w, b_y + b_h

    # Max distance from center to each boundary (half-width, half-height)
    half_crop_w = min(center_x - bx1, bx2 - center_x)
    half_crop_h = min(center_y - by1, by2 - center_y)

    # Use integer values (optional but usually required for indexing)
    half_crop_w = int(half_crop_w)
    half_crop_h = int(half_crop_h)

    x1 = center_x - half_crop_w
    x2 = center_x + half_crop_w
    y1 = center_y - half_crop_h
    y2 = center_y + half_crop_h

    return half_crop_w, half_crop_h


# Defines area of the rectangular region used to crop the well for all imaging sessions
def get_crop_params_from_first_bf(image, crop_percent=0.05, debug=True):
    # Convert to 8-bit if needed
    if image.dtype != np.uint8:
        image = (image / image.max() * 255).astype(np.uint8)

    # Step 1: Binary thresholding
    blur = cv2.GaussianBlur(image, (7, 7), sigmaX=1)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Morphological closing to smooth gaps/holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 3: Blur and re-threshold to soften edges
    blurred = cv2.GaussianBlur(closed, (35, 35), 0)
    _, final_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Find contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found.")

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    center_x = x + w // 2
    center_y = y + h // 2

    # Step 5: DEFINE MAX BOUNDARIES FOR CROPPING
    pad_x = int(w * crop_percent)
    pad_y = int(h * crop_percent)

    b_x = max(x + pad_x, 0)
    b_y = max(y + pad_y, 0)
    b_w = min(w - pad_x, image.shape[1])
    b_h = min(h - pad_y, image.shape[0])

    half_crop_w, half_crop_h = get_crop_box_within_bounds(center_x, center_y,
                                                          b_x, b_y,
                                                          b_w, b_h)
    return half_crop_w, half_crop_h



# detect center of the well using bf image
def detect_well_roi(image, crop_params, params_dict, debug=True, title=None, use_markers=True):
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    film_version = params_dict.get("template", "v1")
    well_width_um = well_width_microns[film_version]
    bridge_width_um = bridge_width_microns[film_version]

    # Step 1: Binary thresholding and morphology
    blur = cv2.GaussianBlur(image, (11, 11), sigmaX=1)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    blurred = cv2.GaussianBlur(closed, (35, 35), 0)
    _, final_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found.")

    # find rectangle coordinates bounding the well
    largest_contour = max(contours, key=cv2.contourArea)
    x_w, y_w, w_w, h_w = cv2.boundingRect(largest_contour)

    # find center of the well
    center_x = x_w + w_w // 2
    center_y = y_w + h_w // 2

    # top left and bottom right corners defining crop bounding box
    half_crop_w, half_crop_h = crop_params
    x1, x2 = center_x - half_crop_w, center_x + half_crop_w
    y1, y2 = center_y - half_crop_h, center_y + half_crop_h

    # pdb.set_trace()
    if use_markers:
        # Crop image using the initial box
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            raise ValueError("Cropped region is empty.")

        # Preprocess cropped image
        cropped_blur = cv2.medianBlur(cropped, 5)

        # Use physical info to estimate circle size in pixels
        try:
            res = params_dict["resolution"]
            half_bridge_width = (bridge_width_um / res) / 2
            half_well_width = (well_width_um / res) / 2
            marker_padding = round((half_well_width + half_bridge_width)/2)
        except KeyError:
            print("Microscope resolution not provided. Using default (0.63 μm/pixel).")
            res = 0.63
            half_bridge_width = (20 / res) / 2
            half_well_width = (51 / res) / 2
            marker_padding = round(half_well_width + half_bridge_width)

        microns_per_pixel = res
        well_diameter_um = well_width_um
        radius_pixels = int((well_diameter_um / microns_per_pixel) / 2)

        # Hough Circle detection parameters
        dp = 1
        param1 = 50   # canny edge detection param
        param2 = 50
        min_dist = radius_pixels * 1.5
        min_radius = int(radius_pixels * 0.85)
        max_radius = int(radius_pixels * 1.15)

        try:
            # Detect circles
            circles = cv2.HoughCircles(
                cropped_blur,
                cv2.HOUGH_GRADIENT,
                dp=dp,
                minDist=min_dist,
                param1=param1,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )
        except Exception as e:
            print(f"{e}")
            print(f"Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")
            sys.exit(1)

        if circles is None:
            raise ValueError("No circles found in cropped region using Hough Transform.")

        circles = np.around(circles[0])

        overlay = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        circular_circles = []

        height, width = cropped.shape

        # filter out perfectly circular circles
        for circle in circles:
            x, y, r = map(int, circle)
            # 1. Skip circles touching frame edge
            if (x - r < 0) or (x + r > width) or (y - r < 0) or (y + r > height):
                if debug:
                    cv2.circle(overlay, (x, y), r, (255, 255, 0), 1)  # yellow outline for skipped
                    cv2.putText(overlay, "Edge", (x + 5, y + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                continue
            # Mask & find contour
            mask = np.zeros_like(cropped, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            result = cv2.bitwise_and(cropped, cropped, mask=mask)

            _, bin_mask = cv2.threshold(result, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            circularity = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    break
                circularity = 4 * np.pi * (area / (perimeter ** 2))

                if 0.85 < circularity < 1.15:
                    circular_circles.append((x, y, r))
                    cv2.circle(overlay, (x, y), r, (0, 255, 0), 10)  # green circle
                    cv2.circle(overlay, (x, y), 2, (0, 255, 0), -1)
                    cv2.putText(
                        overlay,
                        f"{r * microns_per_pixel:.1f}µm, C={circularity:.2f}",
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA
                    )
                else:
                    cv2.circle(overlay, (x, y), r, (255, 0, 0), 15)  # blue for rejected
                    cv2.circle(overlay, (x, y), 2, (255, 0, 0), -1)
                    cv2.putText(
                        overlay,
                        f"{r * microns_per_pixel:.1f}µm, C={circularity:.2f}",
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA
                    )
                break

        if len(circular_circles) < 2:
            raise ValueError("Less than two circular markers found after filtering.")


        # interpolate missing or undetected circles
        well_width_pixels = well_width_um/microns_per_pixel
        bridge_width_pixels = bridge_width_um/ microns_per_pixel
        dx_markers = (well_width_pixels + bridge_width_pixels)*10
        dy_markers =(well_width_pixels + bridge_width_pixels)*10

        centers = [(circle[0], circle[1]) for circle in circular_circles]
        interpolated_circles = interpolate_missing_points(centers, dx_markers, dy_markers, cropped.shape)

        # Use bounding box of detected circular markers
        max_y, max_x = cropped.shape
        xs = [c[0] for c in circular_circles]
        ys = [c[1] for c in circular_circles]
        marker_x1, marker_x2 = min(xs), max(xs)
        marker_y1, marker_y2 = min(ys), max(ys)

        # Optional debug display
        if debug:
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            cv2.circle(overlay_rgb, (marker_x1- marker_padding, marker_y1- marker_padding), 10, (0, 255, 255), -1)  # green circle
            cv2.circle(overlay_rgb, (marker_x2+ marker_padding, marker_y2+ marker_padding), 10, (0, 255, 255), -1)
            cv2.rectangle(overlay_rgb, (marker_x1- marker_padding, marker_y1- marker_padding), (marker_x2+ marker_padding, marker_y2+ marker_padding), (255, 255, 0), 15)
            plt.figure(figsize=(12, 12))
            plt.imshow(overlay_rgb)
            plt.title(title or "Filtered Circles with Annotations")
            plt.axis("off")
            plt.show()


        # Use bounding box of detected circular markers
        xs = [c[0] for c in interpolated_circles if (abs(c[0] - 0) > well_width_pixels and abs(c[0] - max_x) > well_width_pixels)]
        ys = [c[1] for c in interpolated_circles if (abs(c[1] - 0) > well_width_pixels and abs(c[1] - max_y) > well_width_pixels)]
        marker_x1, marker_x2 = min(xs).item(), max(xs).item()
        marker_y1, marker_y2 = min(ys).item(), max(ys).item()

        # Optional debug display
        if debug:
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            for circle in interpolated_circles:
                cv2.circle(overlay_rgb, (circle[0], circle[1]), int(well_width_pixels/2), (0, 255, 0), 10)
            cv2.circle(overlay_rgb, (marker_x1 - marker_padding, marker_y1 - marker_padding), 10, (0, 255, 255), -1)  # green circle
            cv2.circle(overlay_rgb, (marker_x2 + marker_padding, marker_y2 + marker_padding), 10, (0, 255, 255), -1)
            cv2.rectangle(overlay_rgb, (marker_x1 - marker_padding, marker_y1 - marker_padding),
                          (marker_x2 + marker_padding, marker_y2 + marker_padding), (255, 255, 0), 15)
            plt.figure(figsize=(12, 12))
            plt.imshow(overlay_rgb)
            plt.title(title or "Filtered Circles with Annotations")
            plt.axis("off")
            plt.show()

        # Convert local cropped coordinates to global image coordinates
        x1_m = x1 + marker_x1
        x2_m = x1 + marker_x2
        y1_m = y1 + marker_y1
        y2_m = y1 + marker_y2

        # apply padding so markers don't get cut
        x1 = min(x1_m, x1_m - marker_padding)
        y1 = min(y1_m, y1_m - marker_padding)
        x2 = max(x2_m, x2_m + marker_padding)
        y2 = max(y2_m, y2_m + marker_padding)

        # x1 = x1_m
        # y1 = y1_m
        # x2 = x2_m
        # y2 = y2_m

        # return crop box coordinate instead of well as top left corner cannot be used as the reference
        return (x1, y1, x2, y2), (x1, y1, x2, y2), (x_w, y_w, x_w + w_w, y_w + h_w)

    # don't use markers
    else:
        if debug:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            radius = 70
            color = (255, 255, 0)  # BGR
            thickness = 5  # Set to -1 to fill the star
            # Mark the reference point on the image
            points = []
            for i in range(10):
                angle = i * (2 * math.pi / 10) - math.pi / 2  # start pointing up
                r = radius if i % 2 == 0 else radius * 0.5
                x = int(x_w + r * math.cos(angle))
                y = int(y_w + r * math.sin(angle))
                points.append([x, y])

            # Convert to np array for OpenCV
            pts = np.array([points], dtype=np.int32)

            # Draw the star to mark reference point
            cv2.polylines(overlay, pts, isClosed=True, color=color, thickness=thickness)
            cv2.putText(overlay, f"(0, 0)", (x_w - 65, y_w - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            # Draw rectangles and save cropped frames

            # cv2.drawContours(overlay, [largest_contour], -1, (0, 0, 255), 10)
            cv2.rectangle(overlay, (x_w, y_w), (x_w + w_w, y_w + h_w), (0, 255, 0), 15)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 15)
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 12))
            plt.imshow(overlay_rgb)
            plt.title(title or "Debug Crop Visualization")
            plt.axis("off")
            plt.show()

        return (x1, y1, x2, y2), (x_w, y_w, x_w + w_w, y_w + h_w), (x_w, y_w, x_w + w_w, y_w + h_w)


def save_frames_from_cropped_stitched_image(roi_stitched, well_cropped_stitched, output_folder, resolution, fname, roi_bbox_coords,
                                            reference_marker_coords, params_dict, debug=False):
    # Settings
    well_dim_real = well_width_microns[params_dict.get("template", "v1")]
    bridge_dist_real = bridge_width_microns[params_dict.get("template", "v1")]
    num_wells_per_frame = params_dict.get("num_wells_per_frame", 11)
    x_roi, y_roi, _, _ = roi_bbox_coords
    x_0, y_0 = reference_marker_coords[0], reference_marker_coords[1]
    file_dtype = well_cropped_stitched.dtype

    # Pixel calculations
    well_dim_in_pixels = well_dim_real / resolution
    bridge_dist_in_pixels = bridge_dist_real / resolution
    frame_length = round((well_dim_in_pixels + bridge_dist_in_pixels) * num_wells_per_frame)

    img_shape = roi_stitched.shape
    n_rows_frames = img_shape[0] // frame_length
    n_cols_frames = img_shape[1] // frame_length

    if not debug:
        os.makedirs(output_folder, exist_ok=True)

    base_fname = os.path.splitext(fname)[0]

    # Convert image to BGR for color drawing
    norm_img = (well_cropped_stitched / np.max(well_cropped_stitched) * 255).astype(np.uint8) if np.max(well_cropped_stitched) > 255 else well_cropped_stitched.astype(np.uint8)
    if len(norm_img.shape) == 2:
        display_img_pix = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
    else:
        display_img_pix = norm_img.copy()

    radius = 40
    color = (255, 255, 0)  # BGR
    thickness = 2  # Set to -1 to fill the star

    # Mark the reference point on the image
    points = []
    for i in range(10):
        angle = i * (2 * math.pi / 10) - math.pi / 2  # start pointing up
        r = radius if i % 2 == 0 else radius * 0.5
        x = int(x_0 + r * math.cos(angle))
        y = int(y_0 + r * math.sin(angle))
        points.append([x, y])

    # Convert to np array for OpenCV
    pts = np.array([points], dtype=np.int32)

    # Draw the star to mark reference point
    cv2.polylines(display_img_pix, pts, isClosed=True, color=color, thickness=thickness)
    cv2.putText(display_img_pix, f"({x_0}, {y_0})", (x_0 - 35, y_0 - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    # Draw rectangles and save cropped frames
    for row in range(n_rows_frames):
        for col in range(n_cols_frames):
            y = row * frame_length
            x = col * frame_length
            if y + frame_length > img_shape[0] or x + frame_length > img_shape[1]:
                continue

            frame = roi_stitched[y:y + frame_length, x:x + frame_length]


            """
            We have global coordinates of the ROI(yellow) and the well(green). To get 
            comparable coordinates across days, we will now use top left corner of the well as (0, 0). 
            To make this adjustment, add x_roi and y_roi to the original crop coordinates and subtract that from the well coordinates.

            This gives true coordinates of each frame in the original uncropped stitched image of the well.
            This will vary across days.
            """
            x_global = x + x_roi
            y_global = y + y_roi

            """
            To get coordinates with the top left corner of the well as (0, 0) or top left marker, subtract x_0 and y_0(coords of the reference 
            points passed) from the global coordinates.
            """
            x_ref_rel = x_global - x_0
            y_ref_rel = y_global - y_0

            # get realtive coords in microns
            x_ref_rel_microns = round(x_ref_rel*resolution, 2)
            y_ref_rel_microns = round(y_ref_rel*resolution, 2)

            # out_fname = f"{base_fname}_{y_ref_rel_microns}_{x_ref_rel_microns}_{row}_{col}.tif"
            out_fname = f"{base_fname}_r{row}_c{col}.tif"
            out_path = output_folder / out_fname

            if not debug:
                tifffile.imwrite(out_path, frame.astype(file_dtype))

            # Draw rectangle and text
            top_left = (x_global, y_global)
            bottom_right = (x_global + frame_length, y_global + frame_length)
            cv2.rectangle(display_img_pix, top_left, bottom_right, color=(0, 255, 255), thickness=2)  # Yellow

            label = f"({row}, {col}) ({y_ref_rel_microns}um, {x_ref_rel_microns}um)"
            cv2.putText(display_img_pix, label, (x_global + 5, y_global + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Save the annotated image
    parts = output_folder.parts
    device_dir = Path(*parts[:7])
    well_dir = Path(*parts[7:9])
    curr_sname = parts[9]
    channel_str = parts[10]
    tracking_results_folder = device_dir / 'results' / 'annotations' / curr_sname
    tracking_results_folder.mkdir(parents=True, exist_ok=True)
    frame_params_pix_path = tracking_results_folder / f"{base_fname}_{channel_str}_stitched_annotation.png"
    if not debug:
        cv2.imwrite(str(frame_params_pix_path), display_img_pix)
    else:
        pass


def crop_and_save(image_path, crop_box, save_path, save=True, verbose=False):
    with TiffFile(image_path) as tif:
        if verbose:
            print(f"Number of pages: {len(tif.pages)} for file {image_path.name}")
        img = tif.pages[0].asarray()

    # Convert to grayscale if image is colored (assumes last dimension = channels)
    if img.ndim == 3 and img.shape[-1] in [3, 4]:  # RGB or RGBA
        if verbose:
            print(f"Converting color image to grayscale (original shape: {img.shape})")
        img = np.mean(img[..., :3], axis=-1).astype(img.dtype)  # Keep bit depth

    # Crop the image
    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        cropped = img[y1:y2, x1:x2]
    else:
        cropped = img
        if verbose:
            print(f"cropped_roi_shape: {cropped.shape}")

    # Save if needed
    if save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tifffile.imwrite(save_path, cropped)
        print(f"Cropped image saved at {save_path}")

    return cropped, img