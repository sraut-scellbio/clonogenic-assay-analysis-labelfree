import csv
import pdb
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import cv2
import math
import numpy as np
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from preprocessing.coeffs.sc_filter_params import area_coeffs
from preprocessing.helpers.processing_utils import disp_img
from processing.estimate_counts_area_ar import count_tpr_simple, count_tpr_optimized

import warnings

from skimage.measure import regionprops

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
# matplotlib.use('Agg')

def draw_contours(img, contours):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    img_clr = np.uint8(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    cv2.drawContours(img_clr, contours, -1, (255, 0, 0), 1)
    return img_clr

# gives counts based on countour analysis
def get_num_cells_in_contour(cropped_well,
                             cnt,
                             cell_line,
                             cnt_count,
                             microns_per_pixel,
                             model=None,
                             debug=False):

    cnt_area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * (cnt_area / (perimeter * perimeter)) if perimeter != 0 else -1

    if len(cnt) >= 3:
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), _ = rect
        x, y, w, h = int(x), int(y), int(w), int(h)
        cnt_ar = w / h if h > 0 else 0

        roi = cropped_well[y:y + h, x:x + w]

        if debug:
            print(f'\nInfo for contour {cnt_count}:')
            print(f'Contour aspect_ratio: {w / h if h > 0 else 0}')
            print(f'Contour area(microns): {cnt_area * microns_per_pixel ** 2}')
            print(f'Contour circularity: {circularity}')
            disp_img(draw_contours(cropped_well, [cnt]))

        if model is not None:

            # Prepare features for model prediction
            feature_vector = [[
                cnt_area * microns_per_pixel ** 2,  # Area in microns
                cnt_ar,  # Aspect Ratio
                circularity  # Circularity
            ]]

            pred_label = model.predict(feature_vector)[0]

            # get category for contour
            cnt_category = 'single' if pred_label == 1 else 'non-single'

            # get/estimate number of cells
            if cnt_category == 'single':
                num_cells_in_cnt = 1
            else:
                num_cells_in_cnt = count_tpr_optimized(cnt_area_in_pixels=cnt_area,
                                                   cnt_ar=cnt_ar,
                                                   microns_per_pixel=microns_per_pixel,
                                                   cell_line=cell_line)

            if debug:
                print(f"\nPredicted contour category: {cnt_category}\n"
                      f"Predicted num cells per contour: {num_cells_in_cnt}\n")
        else:
            num_cells_in_cnt = count_tpr_optimized(cnt_area_in_pixels=cnt_area,
                                                   cnt_ar=cnt_ar,
                                                   microns_per_pixel=microns_per_pixel,
                                                   cell_line=cell_line)
            cnt_category = 'single' if num_cells_in_cnt == 1 else 'non-single'

            if debug:
                print(f"\nPredicted contour category: {cnt_category}\n"
                      f"Predicted num cells per contour: {num_cells_in_cnt}\n")

        # 'Center X', 'Center Y', 'Area in microns', 'Aspect Ratio', 'Circularity', 'Cnt Category', 'Pred Num Cells'
        return [x, y, cnt_area * microns_per_pixel ** 2, w / h if h > 0 else 0, circularity, cnt_category, num_cells_in_cnt]
    else:
        # print("Detected contour has less than 3 points.")
        return None


def save_cell_counts_combined(
    fluo_img_path,
    lf_img_path,
    well_locs,
    well_width_pixels,
    cell_line,
    microns_per_pixel,
    avg_sc_intensity=None,
    model=None,
    cnt_count=0,
    out_dir=None,
    debug=False,
    save_cropped_wells=False,
    write_results=False,
    clone_threshold=40,
    cluster_threshold=10
):


    min_area_in_pixels = area_coeffs[cell_line]['min'] / (microns_per_pixel ** 2)
    avg_area_in_pixels = area_coeffs[cell_line]['mu'] / (microns_per_pixel ** 2)

    # this is the maximum number of cells that can be predicted using area in a plane
    well_area = (well_width_pixels-10) * 2
    packing_factor = math.pi / (2 * math.sqrt(3))  # hexagona
    max_cells_per_plane = well_area*packing_factor/avg_area_in_pixels

    # Load images
    bf_img = cv2.imread(lf_img_path, cv2.IMREAD_GRAYSCALE)
    bf_img16 = cv2.imread(lf_img_path, cv2.IMREAD_UNCHANGED)
    fluo_img = cv2.imread(fluo_img_path, cv2.IMREAD_GRAYSCALE)
    fluo_img16 = cv2.imread(fluo_img_path, cv2.IMREAD_UNCHANGED)

    if fluo_img is None:
        # raise ValueError(f"Image not found or unreadable: {fluo_img_path}")
        return

    half_width = round(well_width_pixels // 2)
    image_name = Path(fluo_img_path).stem

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_filename_combined = out_dir / f"{image_name}_counts.csv"
    else:
        csv_filename_combined = f"{image_name}_counts.csv"

    combined_data = []

    for (x_w, y_w) in well_locs:

        well_class_shape = None
        well_class_intensity = None

        # Crop images
        x1 = max(x_w - half_width, 0)
        y1 = max(y_w - half_width, 0)
        x2 = min(x_w + half_width, fluo_img.shape[1])
        y2 = min(y_w + half_width, fluo_img.shape[0])

        cropped_fl8 = fluo_img[y1:y2, x1:x2]
        cropped_bf8 = bf_img[y1:y2, x1:x2]
        cropped_bf16 = bf_img16[y1:y2, x1:x2]
        cropped_fl16 = fluo_img16[y1:y2, x1:x2]

        pixels_over_thresh = np.sum(cropped_fl8 > 25)
        # print(f"Pixels over thresh 30: {pixels_over_thresh}")

        num_cells_shape = 0
        num_cells_intensity = 0

        cnt_categories_for_well = []

        if pixels_over_thresh >= min_area_in_pixels:

            # === Non-intensity shape-based workflow ===
            blur = cv2.GaussianBlur(cropped_fl8, (3, 3), 0)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                res = get_num_cells_in_contour(cropped_fl8, cnt, cell_line, cnt_count, microns_per_pixel, model, debug=False)
                if res is not None:
                    num_cells_shape += int(round(res[-1]))
                    cnt_categories_for_well.append(res[-2])

            if avg_sc_intensity is not None:
                # === Intensity-based workflow ===
                dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask = cv2.dilate(binary, dilation_kernel)
                masked_fluo = cv2.bitwise_and(cropped_fl16, cropped_fl16, mask=mask)
                well_intensity_sum = np.sum(masked_fluo)
                num_cells_intensity = int(round(well_intensity_sum / avg_sc_intensity))
            else:
                num_cells_intensity = None

        if num_cells_shape is None:
            well_class_shape = None
        else:
            # Classify well for shape based workflow
            if num_cells_shape == 0:
                well_class_shape = 'empty'
            elif num_cells_shape == 1:
                uniq_categories = set(cnt_categories_for_well)
                if len(uniq_categories) == 1 and list(uniq_categories)[0] == 'single':
                    well_class_shape = 'single'
                else:
                    well_class_shape = 'unknown'
                    num_cells_shape = -1
            elif num_cells_shape == 2:
                well_class_shape = 'dublets'
            elif num_cells_shape == 3:
                well_class_shape = 'triplets'
            elif num_cells_shape == 4:
                well_class_shape = '4'
            # elif num_cells_shape >= clone_threshold:
            #     well_class_shape = 'colonies'
            elif num_cells_shape >= cluster_threshold:
                well_class_shape = '10+'
            elif num_cells_shape >= 5:
                well_class_shape = '5+'
            else:
             well_class_shape = 'unknown'

        if num_cells_intensity is None:
            well_class_intensity = None
        else:
            # Classify well for intensity based workflow
            if num_cells_intensity == 0:
                well_class_intensity = 'empty'
            elif num_cells_intensity == 1:
                well_class_intensity = 'single'
            elif num_cells_intensity == 2:
                well_class_intensity = 'dublets'
            elif num_cells_intensity == 3:
                well_class_intensity = 'triplets'
            elif num_cells_intensity == 4:
                well_class_intensity = '4'
            # elif num_cells_intensity >= clone_threshold:
            #      well_class_intensity = 'colonies'
            elif num_cells_intensity >= cluster_threshold:
                well_class_intensity = '10+'
            elif num_cells_intensity >= 5:
                well_class_intensity = '5+'
            else:
                well_class_intensity = 'unknown'
                num_cells_intensity = -1

        if debug:
            print(f"[{x_w}, {y_w}] shape-based: {num_cells_shape}, intensity: {num_cells_intensity}")
            print(f"[{x_w}, {y_w}] shape-based: {well_class_shape}, intensity: {well_class_intensity}\n")
            disp_img(cropped_bf8, cropped_fl8, title="original cropped_fl8")


        combined_data.append([
            x_w, y_w,
            num_cells_shape,
            well_class_shape,
            num_cells_intensity,
            well_class_intensity
        ])

        if out_dir is not None and save_cropped_wells:
            base_name = Path(lf_img_path).stem
            class_dir_shape = out_dir / "shape_based"/ well_class_shape
            class_dir_shape.mkdir(parents=True, exist_ok=True)
            # save images for shape based workflow
            out_path_lf_shape = class_dir_shape / f"{base_name}_x{x_w}_y{y_w}_lf_count{num_cells_shape}.tif"
            out_path_lab_shape = class_dir_shape / f"{base_name}_x{x_w}_y{y_w}_fl_count{num_cells_shape}.tif"
            cv2.imwrite(str(out_path_lf_shape), cropped_bf16)
            cv2.imwrite(str(out_path_lab_shape), cropped_fl16)

            if avg_sc_intensity:
                # save images for intensity based workflow
                class_dir_int = out_dir / "int_based"/ well_class_intensity
                class_dir_int.mkdir(parents=True, exist_ok=True)
                out_path_lf_int = class_dir_int / f"{base_name}_x{x_w}_y{y_w}_lf_count{num_cells_intensity}.tif"
                out_path_lab_int = class_dir_int / f"{base_name}_x{x_w}_y{y_w}_fl_count{num_cells_intensity}.tif"
                cv2.imwrite(str(out_path_lf_int), cropped_bf16)
                cv2.imwrite(str(out_path_lab_int), cropped_fl16)

    if write_results:
        with open(csv_filename_combined, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Well Center X', 'Well Center Y', 'Model-Based Count', 'Model-Based Category' ,'Intensity-Based Count',
                             'Intensity-Based Category'])
            writer.writerows(combined_data)

    return combined_data


def save_cell_counts_labelfree(
    lf_model,
    flow_threshold,
    cellprob_threshold,
    tile_norm_blocksize,
    lf_img_path: str,
    well_locs: List[Tuple[int, int]],
    cell_line,
    microns_per_pixel,
    well_width_pixels: int,
    out_dir=None,
    write_results=False,
    save_masks_for_training=True,
    save_flows=True,
    debug=False
):
    if lf_model is None:
        print(f"Label-free segmentation model has not been provided.")
        return

    min_area_in_pixels = area_coeffs[cell_line]['min'] / (microns_per_pixel ** 2)
    avg_area_in_pixels = area_coeffs[cell_line]['mu'] / (microns_per_pixel ** 2)

    # Load image
    bf_img = cv2.imread(str(lf_img_path), cv2.IMREAD_GRAYSCALE)
    if bf_img is None:
        raise ValueError(f"Image not found or unreadable: {lf_img_path}")

    image_name = Path(lf_img_path).stem
    out_dir = Path(out_dir) if out_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_filename_combined = out_dir / f"{image_name}_counts.csv"

    half_width = round(well_width_pixels // 2)
    well_area = well_width_pixels**2

    def process_crop(well_coord):
        x_w, y_w = well_coord
        x1, y1 = max(x_w - half_width, 0), max(y_w - half_width, 0)
        x2, y2 = min(x_w + half_width, bf_img.shape[1]), min(y_w + half_width, bf_img.shape[0])
        cropped = bf_img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_LINEAR)
        return x_w, y_w, cropped, resized

    # Use ThreadPoolExecutor for I/O-safe parallelism
    with ThreadPoolExecutor(max_workers=8) as executor:
        crop_meta = list(executor.map(lambda coord: process_crop(coord), well_locs))

    crops = [resized for (_, _, _, resized) in crop_meta]

    print(f"Detecting cells for frame {lf_img_path} with cellpose.")
    masks_list, flows_list, _ = lf_model.eval(
        crops,
        batch_size=64,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize}
    )

    def process_well(i):
        x_w, y_w, cropped, resized = crop_meta[i]
        mask = masks_list[i]
        flow = flows_list[i]


        resize_factor = resized.size / cropped.size
        circularity = None
        if mask is not None and np.max(mask) > 0:
            props = regionprops(mask)
            unknown_obj = False

            num_cells = 0

            for prop in props:
                obj_type, count = count_tpr_simple(prop, avg_area_in_pixels, resize_factor, well_area)

                # update count and object type
                num_cells += count
                unknown_obj = unknown_obj or obj_type

            if num_cells == 0:
                if not unknown_obj:
                    label = 'empty'  # debris or all masks too irregular
                    num_cells = 0
                else:
                    label = 'unknown'  # debris or all masks too irregular
                    num_cells = -1
            else:
                # classification logic
                if num_cells == 1:
                    if not unknown_obj:
                        label = 'single'  # debris or all masks too irregular
                        num_cells = 1
                    else:
                        label = 'unknown'  # debris or all masks too irregular
                        num_cells = -1
                elif num_cells == 2:
                    label = 'dublets'
                elif num_cells == 3:
                    label = 'triplets'
                elif num_cells == 4:
                    label = '4'
                elif num_cells >= 10:
                    label = '10+'
                elif num_cells >= 5:
                    label = '5+'
                else:
                    label = 'unknown'
                    num_cells = -1
        else:
            label = 'empty'
            num_cells = 0

        results = [x_w, y_w, num_cells, label]

        # Save training data and flows
        if save_masks_for_training or save_flows:
            if label in ["empty", "single", "unknown"]:
                class_dir = out_dir / label
            else:
                class_dir = out_dir / "non-single"
            if save_masks_for_training:
                img_out_dir = class_dir / "images"
                mask_out_dir = class_dir / "masks"
                img_out_dir.mkdir(parents=True, exist_ok=True)
                mask_out_dir.mkdir(parents=True, exist_ok=True)

                base_name = f"{image_name}_x{x_w}_y{y_w}"
                cv2.imwrite(str(img_out_dir / f"{base_name}.tif"), resized)
                cv2.imwrite(str(mask_out_dir / f"{base_name}_mask.tif"), mask.astype(np.uint16))

            if save_flows:
                flows_dir = class_dir / "flows"
                flows_dir.mkdir(parents=True, exist_ok=True)

                if circularity is not None:
                    fig_path = flows_dir / f"{image_name}_x{x_w}_y{y_w}_{circularity:.2f}.png"
                else:
                    fig_path = flows_dir / f"{image_name}_x{x_w}_y{y_w}.png"

                # use matplotlib for debugging
                if debug:
                    prob_map = flow[0]
                    flow_map = flow[1]

                    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
                    axes[0].imshow(resized, cmap='gray')
                    axes[0].set_title('Brightfield')
                    axes[0].axis('off')

                    # Mask image or blank fallback
                    if mask is not None and np.any(mask):
                        axes[1].imshow(mask, cmap='nipy_spectral')
                        axes[1].set_title(f'Mask (n={num_cells})')
                    else:
                        axes[1].imshow(np.zeros_like(resized), cmap='gray')
                        axes[1].set_title("Mask: None")

                    axes[1].axis('off')

                    # Cell probability map
                    prob_map = np.nan_to_num(prob_map)
                    axes[2].imshow(prob_map, cmap='viridis')
                    axes[2].set_title('Cell Prob Map')
                    axes[2].axis('off')

                    # Flow field
                    step = 5
                    h, w = resized.shape
                    Y, X = np.mgrid[0:h:step, 0:w:step]
                    U = np.nan_to_num(flow_map[1][0:h:step, 0:w:step])
                    V = np.nan_to_num(flow_map[0][0:h:step, 0:w:step])

                    axes[3].imshow(resized, cmap='gray')
                    axes[3].quiver(X, Y, U, -V, color='red', scale=20)
                    axes[3].set_title('Flow Field')
                    axes[3].axis('off')

                    plt.tight_layout()
                    try:
                        plt.show()
                    except Exception as e:
                        print(f"Error saving flow figure at {fig_path}: {e}")
                    plt.close(fig)

                else:
                    try:
                        def visualize_instance_mask(mask, colormap=cv2.COLORMAP_JET):
                            """
                            Visualize instance mask with distinct colors and black background.
                            """
                            vis = np.zeros((*mask.shape, 3), dtype=np.uint8)  # RGB black canvas
                            unique_ids = np.unique(mask)
                            for uid in unique_ids:
                                if uid == 0:
                                    continue  # background stays black
                                color = tuple(int(c) for c in np.random.randint(50, 255, size=3))  # bright color
                                vis[mask == uid] = color
                            return vis


                        # Brightfield image (normalized, grayscale -> RGB)
                        norm_bf = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        bf_rgb = cv2.cvtColor(norm_bf, cv2.COLOR_GRAY2RGB)

                        # Instance mask visualization (independent, no overlay)
                        mask_rgb = visualize_instance_mask(mask)

                        # Use Cellpose’s own probability map (already meaningful)
                        prob_map = np.nan_to_num(flow[0])  # shape: (H, W), dtype: float32 or float64
                        prob_norm = cv2.normalize(prob_map, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)
                        colormap = cm.get_cmap('jet')  # Or 'rainbow'
                        colored_map = colormap(prob_norm)
                        prob_map = (colored_map[:, :, :3] * 255).astype(np.uint8)
                        prob_rgb = cv2.cvtColor(prob_map, cv2.COLOR_RGB2BGR)

                        # Flow field visualization (on top of brightfield)
                        flow_rgb = bf_rgb.copy()
                        step = 10
                        h, w = resized.shape
                        for y in range(0, h, step):
                            for x in range(0, w, step):
                                dx = int(flow[1][1][y, x] * 5)
                                dy = int(flow[1][0][y, x] * 5)
                                pt1 = (x, y)
                                pt2 = (x + dx, y + dy)
                                cv2.arrowedLine(flow_rgb, pt1, pt2, (255, 0, 0), 1, tipLength=0.3)  # Red arrows in RGB

                        def resize_rgb(img):
                            if img.shape[:2] != (256, 256):
                                img = cv2.resize(img, (256, 256))
                            return img

                        final_vis = np.hstack([resize_rgb(bf_rgb), resize_rgb(mask_rgb)])
                        # stack1 = np.hstack([resize_rgb(bf_rgb), resize_rgb(mask_rgb)])
                        # stack2 = np.hstack([resize_rgb(prob_rgb), resize_rgb(flow_rgb)])
                        # final_vis = np.vstack([stack1, stack2])

                        # Save final RGB visualization
                        if final_vis.ndim == 3 and final_vis.shape[2] == 3:
                            # Already 3-channel image; OpenCV expects BGR when saving
                            final_bgr = cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR)
                        else:
                            # Single-channel or unexpected shape
                            final_bgr = final_vis

                        cv2.imwrite(str(fig_path), final_bgr)

                    except Exception as e:
                        print(f"Error saving flow figure at {fig_path}: {e}")

        return results

    combined_data = []
    for i in range(len(crop_meta)):
        result = process_well(i)
        combined_data.append(result)

    # Save CSV results
    if write_results:
        with open(csv_filename_combined, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Well Center X', 'Well Center Y', 'Model-Based Count', 'Model-Based Category'])
            writer.writerows(combined_data)

    return combined_data


# get per frame intensity
def get_cleaned_intensity_value(
    fluo_img_path,
    background_profile,
    debug=False
)->float:
    # 1) load raw 16‑bit
    img16 = cv2.imread(str(fluo_img_path), cv2.IMREAD_UNCHANGED)
    if img16 is None:
        raise ValueError(f"Cannot read {fluo_img_path}")
    # ensure we got a single‐channel image
    if img16.dtype not in (np.uint16, np.uint8):
        raise ValueError(f"Unexpected dtype {img16.dtype} for {fluo_img_path}")

    # 2) subtract & 3) clip
    img_f = img16.astype(np.float32) - background_profile
    img_f[img_f < 0] = 0

    if debug:
        disp_img(img16, img_f)

    # 4) calculate intesnity for background corrected image
    intensity_sum = np.sum(img_f)

    return intensity_sum