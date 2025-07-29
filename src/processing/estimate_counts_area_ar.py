import cv2
import numpy as np
from preprocessing.coeffs.sc_filter_params import area_coeffs, aspect_ratio_coeffs, distr_coeffs

# this workflow uses parameters that aim to lower false positives for singles
def count_fpr_optimized(cnt_area, cnt_ar, microns_per_pixel, cell_line):
    # THSES VALUES ARE POST OUTLIER REMOVAL
    # define area filter parameters for single cells
    mu_area = area_coeffs[cell_line]["mu"] / (microns_per_pixel ** 2)
    std_area = area_coeffs[cell_line]["std"] / (microns_per_pixel ** 2)


    # define aspect ratio filter parmaters for single cells
    mu_ar = aspect_ratio_coeffs[cell_line]["mu"]
    std_ar = aspect_ratio_coeffs[cell_line]["std"]
    min_ar = aspect_ratio_coeffs[cell_line]["min"]
    max_ar = aspect_ratio_coeffs[cell_line]["max"]

    area_coeff = distr_coeffs[cell_line]["area_coeff"]


    area_range1 = [mu_area - area_coeff*std_area, mu_area + area_coeff*std_area]

    # if area matches average single cell area
    if area_range1[0] <= cnt_area <= area_range1[1]:
        if  min_ar <= cnt_ar <= max_ar:
            num_cells_in_contour = 1
            print("Workflow1: Single cell detected. Count increased by 1.")
        elif 0.25 < cnt_ar < 4:
            num_cells_in_contour = round(cnt_area / mu_area)
            print(f"Workflow1: Contour fits average sc area + min-max aspact ratio criteria. Predicted cells for contour:"
                  f" {num_cells_in_contour}")
        else:
            num_cells_in_contour = -255
            print(f"Workflow1: Contour fits average sc area but aspect ratio similar to relfection. Predicted cells for contour:"
                  f" {num_cells_in_contour}")

    # area larger than average single cell
    elif cnt_area > area_range1[1]:
        if mu_ar - 4.5 * std_ar <= cnt_ar <= mu_ar - 4.5 * std_ar: # non-crazy ar
            num_cells_in_contour = round(cnt_area / mu_area)
            print(f"Workflow1: Contour area larger than average sc area. Predicted cells for contour:"
                  f" {num_cells_in_contour}")
        else:
            num_cells_in_contour = -255
            print(f"Workflow1: Contour area larger than average sc area but aspect ratio similar to relfection. Predicted cells for contour:"
                  f" {num_cells_in_contour}")


    # area smaller than average single cell
    else:
        if cnt_area >= mu_area - 2*std_area: # area within 2 stds
            if mu_ar - 1.5 * std_ar <= cnt_ar <= mu_ar - 1.5 * std_ar: #circular
                num_cells_in_contour = 1
                print(f"Workflow1: Area smller than average single cell. Aspect ratio shows round obj. Single cell detected.")
            else:
                num_cells_in_contour = -255
                print(f"Workflow1: Area smller than average single cell. Aspect ratio shows non-round obj. Val set to -255.")
        else:
            num_cells_in_contour = -255
            print(f"Workflow1: Area smller than average single cell. Value set to negative 255. Val set to -255.")

    return num_cells_in_contour


# this workflow uses parameters that aim to also increase true positives and acc
def count_tpr_optimized(cnt_area_in_pixels, cnt_ar, microns_per_pixel, cell_line, verbose=False):
    # THSES VALUES ARE POST OUTLIER REMOVAL
    # define area filter parameters for single cells
    mu_area = area_coeffs[cell_line]["mu"] / (microns_per_pixel ** 2)
    std_area = area_coeffs[cell_line]["std"] / (microns_per_pixel ** 2)

    # define aspect ratio filter parmaters for single cells
    mu_ar = aspect_ratio_coeffs[cell_line]["mu"]
    std_ar = aspect_ratio_coeffs[cell_line]["std"]
    min_ar = aspect_ratio_coeffs[cell_line]["min"]
    max_ar = aspect_ratio_coeffs[cell_line]["max"]

    area_coeff = distr_coeffs[cell_line]["area_coeff"]

    area_range1 = [mu_area - area_coeff * std_area, mu_area + area_coeff * std_area]

    # if area matches average single cell area
    if area_range1[0] <= cnt_area_in_pixels <= area_range1[1]:
        if min_ar <= cnt_ar <= max_ar:
            num_cells_in_contour = 1
            if verbose: print("Workflow1: Single cell detected. Count increased by 1.")
        elif 0.25 < cnt_ar < 4:
            num_cells_in_contour = round(cnt_area_in_pixels / mu_area)
            if verbose: print(f"Workflow1: Contour fits average sc area + min-max aspact ratio criteria. Predicted cells for contour:"
                  f" {num_cells_in_contour}")
        else:
            num_cells_in_contour = 0
            if verbose: print(f"Workflow1: Contour fits average sc area but aspect ratio similar to relfection. Predicted cells for contour:"
                  f" {num_cells_in_contour}")

    # area larger than average single cell
    elif cnt_area_in_pixels > area_range1[1]:
        if mu_ar - std_ar <= cnt_ar <= mu_ar - std_ar:
            num_cells_in_contour = 1
            if verbose: print(f"Workflow1: Contour area larger than average sc area but aspect ratio close to 1. Predicted LARGE single cell.")
        elif 0.15 < cnt_ar < 3.5:  # non-crazy ar
            num_cells_in_contour = round(cnt_area_in_pixels / mu_area)
            if verbose: print(f"Workflow1: Contour area larger than average sc area but aspect ratio not closr to 1. Predicted cells for contour:"
                  f" {num_cells_in_contour}")
        else:
            num_cells_in_contour = 0
            if verbose: print(f"Workflow1: Contour area larger than average sc area but aspect ratio similar to relfection. Predicted cells for "
                            f"contour:"
                  f" {num_cells_in_contour}")


    # area smaller than average single cell
    else:
        if cnt_area_in_pixels >= mu_area - 2 * std_area:  # area within 2 stds
            if mu_ar - 1.5 * std_ar <= cnt_ar <= mu_ar - 1.5 * std_ar:  # circular
                num_cells_in_contour = 1
                if verbose: print(f"Workflow1: Area smller than average single cell. Aspect ratio shows round obj. Single cell detected.")
            else:
                num_cells_in_contour = 0
                if verbose: print(f"Workflow1: Area smller than average single cell. Aspect ratio shows non-round obj. Val set to -255.")
        else:
            num_cells_in_contour = 0
            if verbose: print(f"Workflow1: Area smller than average single cell. Value set to negative 255. Val set to -255.")

    return num_cells_in_contour


# simplified counting for individual masks for labelfree
def count_tpr_simple(prop, avg_area_in_pixels, resize_factor, well_area_in_pixels, verbose=False):
    unknown_obj = False
    area = prop.area
    perimeter = prop.perimeter if prop.perimeter > 0 else 1
    circularity = 4 * np.pi * area / (perimeter ** 2)

    # calculate aspect ratio
    # convert binary mask to contour for this object
    mask = prop.image.astype(np.uint8)  # binary mask for the object
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and len(contours[0]) >= 5:  # OpenCV requires at least 5 points for a valid rot rect
        rect = cv2.minAreaRect(contours[0])
        (center), (width, height), angle = rect
        cnt_ar = max(width, height) / min(width, height) if min(width, height) > 0 else 0
    else:
        cnt_ar = 0  # fallback if contour fails

    min_area =  0.55 * avg_area_in_pixels*resize_factor
    max_area = 0.35 * well_area_in_pixels*resize_factor

    if circularity >= 0.6:
        # Rule 1: Too large area (> 35% of well)
        if area > max_area:
            if verbose: print("Area too large. Count = 0.")
            return unknown_obj, 0

        # Rule 2: Acceptable area
        elif min_area <= area <= max_area:

            # acceptable aspect ratio for single
            if 0.6 <= cnt_ar <= 1.5:
                if verbose: print("Acceptable area, but high AR. Count = 2.")
                return unknown_obj, 1

            # high ar might be a dublet
            else:
                if verbose: print("Acceptable area and AR. Count = 1.")
                return unknown_obj, 2

        # Rule 3: Too small area
        else:
            if verbose: print("Area too small. Count = 0")
            return unknown_obj, 0

    # non-circular object
    else:
        # non-small unknown object
        if area > min_area:
            unknown_obj = True

        # small unknown object might be debris or empty
        else:
            unknown_obj = False

        return unknown_obj, 0

