import random

import cv2
import numpy as np

from preprocessing.coeffs.sc_filter_params import area_coeffs
from preprocessing.filetypes import image_types
from preprocessing.helpers.io_utils import disp_img
from processing.well_identification import get_well_locations
from processing.count_cells import get_num_cells_in_contour

def get_avg_single_cell_intensity(lf_dir,
                              fluo_dir,
                              template,
                              microns_per_pixel,
                              cell_line,
                              well_width_pixels,
                              bridge_width_pixels,
                              model,
                              debug=True):

    # step1: randomly select specific number of images to detect_singles
    sample_size = 35
    all_images = [f for f in lf_dir.iterdir() if f.suffix.lower() in image_types.VALID_IMAGE_EXTENSIONS]
    if len(all_images) < sample_size:
        print(f"Only {len(all_images)} images found; returning all.")
        return all_images

    sample_images = random.sample(all_images, sample_size)

    detected_singles = 0
    total_intensity = 0
    for i, lf_img_path in enumerate(sample_images):
        img_name = lf_img_path.name
        well_locs, _ = get_well_locations(lf_img_path, well_width_pixels, bridge_width_pixels,
                                          template, microns_per_pixel, debug_well_detection=False)

        # no wells detected
        if len(well_locs) == 0:
            print(f"No wells detected for this frame.")
            continue

        else:
            fluo_img_path = fluo_dir / img_name

        min_area_in_pixels = area_coeffs[cell_line]['min'] / (microns_per_pixel ** 2)

        # read images
        bf_img8 = cv2.imread(lf_img_path, cv2.IMREAD_GRAYSCALE)
        fluo_img8 = cv2.imread(fluo_img_path, cv2.IMREAD_GRAYSCALE)
        fluo_img16 = cv2.imread(fluo_img_path, cv2.IMREAD_UNCHANGED)

        if fluo_img8 is None:
            # raise ValueError(f"Image not found or unreadable: {fluo_img_path}")
            continue

        half_width = round(well_width_pixels // 2)


        # for each well location
        for (x_w, y_w) in well_locs:

            x1 = max(x_w - half_width, 0)
            y1 = max(y_w - half_width, 0)
            x2 = min(x_w + half_width, fluo_img8.shape[1])
            y2 = min(y_w + half_width, fluo_img8.shape[0])

            cropped_fl8 = fluo_img8[y1:y2, x1:x2]
            cropped_fl16 = fluo_img16[y1:y2, x1:x2]
            cropped_bf = bf_img8[y1:y2, x1:x2]

            pixels_over_thresh = np.sum(cropped_fl8 > 30)
            # print(f"Pixels over thresh 30: {pixels_over_thresh}")

            if pixels_over_thresh >= min_area_in_pixels:
                if debug:
                    disp_img(cropped_bf, cropped_fl8, title="original cropped_fl8")
                blur = cv2.GaussianBlur(cropped_fl8, (3, 3), 0)
                _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # for each contour identified
                for i, cnt in enumerate(contours):
                    res = get_num_cells_in_contour(cropped_fl8,
                                                   cnt,
                                                   cell_line,
                                                   0,
                                                   microns_per_pixel,
                                                   model,
                                                   debug)

                    if res is not None:
                        cnt_category = res[-2]
                        cnt_count = res[-1]
                        if cnt_category == 'single' or cnt_count == 1:

                            # Create empty mask for the cropped region
                            mask = np.zeros(cropped_fl8.shape, dtype=np.uint8)

                            #  Draw filled contour on the mask
                            cv2.drawContours(mask, [cnt], -1, color=255, thickness=-1)  # filled mask

                            # Optional dilation (e.g., 3x3 elliptical kernel)
                            dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                            mask = cv2.dilate(mask, dilation_kernel)

                            # Apply mask to cropped_fl16 fluorescent image
                            masked_fluo = cv2.bitwise_and(cropped_fl16, cropped_fl16, mask=mask)

                            # Calculate sum of intensities
                            intensity_sum = np.sum(masked_fluo)
                            detected_singles += 1
                            total_intensity += intensity_sum

                            if debug:
                                print(f"Sum of intensities in dilated 'single' cell contour: {intensity_sum}")
                                disp_img(masked_fluo)

                        else: # non-single contour
                            pass

                    else: # if result none
                        pass

            else: # empty well
                pass

    # calculate average single cell intensity
    average_sc_intensity = total_intensity/detected_singles
    print(f"Total singles detected for well {lf_dir.parents[1].name}: {detected_singles}")
    print(f"Average singlecell intensity for well {lf_dir.parents[1].name}: {average_sc_intensity}")
    return average_sc_intensity