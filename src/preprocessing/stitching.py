import os
import cv2
import json
import numpy as np
from pathlib import Path

from matplotlib import pyplot as plt
from natsort import natsorted
from scipy.spatial import distance
from scipy.ndimage import maximum_filter


def load_images_natsorted(folder_path):
    paths = natsorted(Path(folder_path).glob("*.tif"))
    images = []
    names = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {p}")
        images.append(img)
        names.append(p.name)
    return images, names


def compute_physical_offsets(n_rows, n_cols, frame_shape, overlap_microns, microns_per_pixel):
    fh, fw = frame_shape
    step_x = int(fw - (overlap_microns / microns_per_pixel))
    step_y = int(fh - (overlap_microns / microns_per_pixel))
    offsets = [(j * step_x, i * step_y) for i in range(n_rows) for j in range(n_cols)]
    return offsets, step_x, step_y


def detect_wells(image, template, threshold=0.6):
    # ==== Match Template ====
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape[:2]

    # ==== Detect Local Maxima ====
    filtered = maximum_filter(result, size=(th, tw))
    local_max_mask = (result == filtered) & (result >= threshold)
    locations = np.where(local_max_mask)
    points = list(zip(*locations[::-1]))  # (x, y)

    # ==== Detected Centers in Image Coordinates ====
    detected_centers = [(x + tw // 2, y + th // 2) for (x, y) in points]
    match_scores = [result[y, x] for (x, y) in points]

    # img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for (x, y), score in zip(detected_centers, match_scores):
    #     cv2.circle(img_color, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
    #     cv2.putText(img_color, f"{score:.2f}", (x + 8, y - 8),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_color)
    # plt.title("Detected Wells (Template Scaled)")
    # plt.axis('off')
    # plt.show()

    return np.array(detected_centers)


def find_shift_from_well_matches(ref_centers, new_centers, axis, max_dist=50):
    if len(ref_centers) == 0 or len(new_centers) == 0:
        return 0
    ref_vals = ref_centers[:, axis]
    new_vals = new_centers[:, axis]
    dists = distance.cdist(ref_vals[:, None], new_vals[:, None])
    i, j = np.unravel_index(np.argmin(dists), dists.shape)
    return ref_vals[i] - new_vals[j] if dists[i, j] < max_dist else 0


def align_and_stitch_with_template(json_path, bf_dir, template_path, microns_per_pixel, overlap_microns, save_path="stitched_template_aligned.tif"):
    with open(json_path, "r") as f:
        grid = json.load(f)
    n_rows, n_cols = grid['montage_grid'][0], grid['montage_grid'][1]

    images, names = load_images_natsorted(bf_dir)
    template = np.load(template_path).astype(np.uint8)
    frame_h, frame_w = images[0].shape

    offsets, step_x, step_y = compute_physical_offsets(n_rows, n_cols, (frame_h, frame_w), overlap_microns, microns_per_pixel)
    canvas_h = step_y * (n_rows - 1) + frame_h
    canvas_w = step_x * (n_cols - 1) + frame_w
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint16)

    prev_row_centers = [[] for _ in range(n_cols)]
    prev_col_centers = [[] for _ in range(n_cols)]

    for idx, (img, (off_x, off_y)) in enumerate(zip(images, offsets)):
        row, col = divmod(idx, n_cols)
        centers = detect_wells(img, template)

        if len(centers) == 0:
            print(f"[WARNING] No wells detected in tile: {names[idx]}")

        dx = dy = 0
        if len(centers) > 0:
            if row > 0 and len(prev_col_centers[col]) > 0:
                dy = find_shift_from_well_matches(prev_col_centers[col], centers, axis=1)
            if col > 0 and len(prev_row_centers[col - 1]) > 0:
                dx = find_shift_from_well_matches(prev_row_centers[col - 1], centers, axis=0)

        x, y = int(off_x + dx), int(off_y + dy)

        # Ensure bounds are within canvas
        x = max(0, min(x, canvas_w - frame_w))
        y = max(0, min(y, canvas_h - frame_h))

        canvas[y:y + frame_h, x:x + frame_w] = img

        if len(centers) > 0:
            global_centers = centers + np.array([x, y])
        else:
            global_centers = np.empty((0, 2))

        prev_row_centers[col] = global_centers
        prev_col_centers[col] = global_centers

    cv2.imwrite(save_path, canvas)
    print(f"✅ Stitched image saved to: {save_path}")


# === USER CONFIGURATION ===
fiji_path = "C:/Users/ShiskaRaut/Fiji.app/ImageJ-win64.exe"
metadata_path = 'data/organized/05_05_25_cytation_SCB_everyday-imaging_C343_None_2025-05-09_12-48-40/C343/original_frames/metadata.json'
bf_path = 'data/organized/05_05_25_cytation_SCB_everyday-imaging_C343_None_2025-05-09_12-48-40/C343/original_frames/D2/day1/bright field'
dapi_path = 'data/organized/05_05_25_cytation_SCB_everyday-imaging_C343_None_2025-05-09_12-48-40/C343/original_frames/D2/day1/dapi'

align_and_stitch_with_template(
    json_path=metadata_path,
    bf_dir=bf_path,
    template_path="templates/final_templates/well/template_v1.npy",
    microns_per_pixel=0.645,
    overlap_microns=100,
    save_path="stitched_output/stitched_template_aligned.tif"
)
