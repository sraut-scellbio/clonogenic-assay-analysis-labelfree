import pdb

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.spatial import KDTree
from preprocessing.coeffs.sc_filter_params import area_coeffs
from preprocessing.helpers.io_utils import disp_img


def suppress_close_points(points, threshold):
    """Merge points closer than threshold using averaging (non-max suppression)."""
    if not points:
        return []
    tree = KDTree(points)
    visited = set()
    suppressed = []

    for i, pt in enumerate(points):
        if i in visited:
            continue
        idxs = tree.query_ball_point(pt, threshold)
        cluster = [points[j] for j in idxs]
        avg_point = np.mean(cluster, axis=0)
        suppressed.append(tuple(avg_point))
        visited.update(idxs)
    return suppressed


def get_vector(pt, tree, est_vector, points_array, est_distance):
    """Use a neighbor to calculate the vector in ±x or ±y direction if available."""
    query_pt = pt + est_vector
    nearby_indices = tree.query_ball_point(query_pt, 0.25 * est_distance)

    if not nearby_indices:
        return None

    candidate_vectors = [
        points_array[i] - pt
        for i in nearby_indices
        if not np.allclose(points_array[i], pt)
    ]

    if candidate_vectors:
        best_vec = min(candidate_vectors, key=lambda v: np.linalg.norm(v - est_vector))
        return tuple(best_vec)

    return None


def interpolate_missing_points(points, dx, dy, image_shape, img_color=None, max_iters=10):
    """
    Interpolates missing well centers from detected points using grid vectors and NMS.
    Ensures original detected points are never overwritten and newly interpolated
    points are filtered based on proximity to existing points.
    """
    est_distance = dx   # estimated distance between wells
    image_h, image_w = image_shape

    # Initial detected points
    original_points = np.array(points)
    final_set = set(tuple(map(int, pt)) for pt in original_points)
    current = original_points

    # Estimate mean horizontal and vertical vectors from original points
    vectors_hz, vectors_vt = [], []
    try:
        tree = KDTree(original_points)
    except Exception:
        print(f"Error in well interpolation. Original points returned.")
        return points

    for pt in original_points:
        pt = np.array(pt)
        vec_hz = get_vector(pt, tree, np.array([dx, 0]), original_points, est_distance)
        if vec_hz:
            vectors_hz.append(vec_hz)

        vec_vt = get_vector(pt, tree, np.array([0, dy]), original_points, est_distance)
        if vec_vt:
            vectors_vt.append(vec_vt)

    mean_vec_hz = np.mean(vectors_hz, axis=0) if vectors_hz else np.array([dx, 0])
    mean_vec_vt = np.mean(vectors_vt, axis=0) if vectors_vt else np.array([0, dy])

    for iteration in range(max_iters):
        new_points = []

        # Build KDTree from final_set to check for proximity-based duplicates
        existing_points_tree = KDTree(np.array(list(final_set)))

        for pt in current:
            pt = np.array(pt)

            # Propose new points in ±x and ±y directions
            candidates = [
                pt + mean_vec_hz,
                pt - mean_vec_hz,
                pt + mean_vec_vt,
                pt - mean_vec_vt
            ]

            for new_pt in candidates:
                x, y = new_pt
                if (
                    0 <= x < image_w and 0 <= y < image_h and
                    not existing_points_tree.query_ball_point(new_pt, r=0.35 * est_distance)
                ):
                    new_points.append(new_pt)

        if not new_points:
            break

        # Suppress overlapping new points
        suppressed_new = suppress_close_points(new_points, threshold=0.35 * est_distance)

        # Round and filter to avoid adding points close to existing final_set
        next_iter_points = []
        for pt in suppressed_new:
            if not existing_points_tree.query_ball_point(pt, r=0.35 * est_distance):
                pt_int = tuple(map(int, pt))
                final_set.add(pt_int)
                next_iter_points.append(pt)
        # for pt in next_iter_points:
        #     pt_int = tuple(map(int, pt))
        #     cv2.circle(img_color, pt_int, radius=3, color=(0, 0, 255), thickness=-1)
        # disp_img(img_color, color=True)

        current = np.array(next_iter_points)

    return np.array(sorted(final_set))


def get_well_locations(img_path, well_width_pixels, bridge_width_pixels,
                       template, microns_per_pixel,
                       match_threshold=0.7, debug_well_detection=False):

    # ==== Define template scaling factor ====
    original_resolution = 0.645  # microns per pixel of the template
    scale = original_resolution / microns_per_pixel  # Scale template to match image resolution

    # ==== Scale the template (not the image) ====
    if scale != 1.0:
        new_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
        template = cv2.resize(template, new_size, interpolation=cv2.INTER_NEAREST)

    # template = template[11:106, 11:106]
    th, tw = template.shape[:2]

    # ==== Load image ====
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # ==== Match Template ====
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    # ==== Detect Local Maxima ====
    filtered = maximum_filter(result, size=(th, tw))
    local_max_mask = (result == filtered) & (result >= match_threshold)
    locations = np.where(local_max_mask)
    points = list(zip(*locations[::-1]))  # (x, y)

    # ==== Detected Centers in Image Coordinates ====
    detected_centers = [(x + tw // 2, y + th // 2) for (x, y) in points]
    match_scores = [result[y, x] for (x, y) in points]

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for (x, y), score in zip(detected_centers, match_scores):
        cv2.circle(img_color, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.putText(img_color, f"({x}, {y})\n{score:.2f}", (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


    dx = well_width_pixels + bridge_width_pixels # spacing between columns in pixels
    dy = well_width_pixels + bridge_width_pixels # spacing between rows in pixel

    if debug_well_detection:
        disp_img(img_color, color=True)

    img_color2 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    if len(detected_centers) > 7:
        interpolated_points = interpolate_missing_points(detected_centers, dx, dy, img_gray.shape, img_color2)

        half_width = well_width_pixels // 2
        # filter valid well locs
        valid_well_locs = [
            (x_w, y_w) for (x_w, y_w) in interpolated_points
            if (
                    x_w - half_width >= 0 and
                    y_w - half_width >= 0 and
                    x_w + half_width < img_gray.shape[1] and
                    y_w + half_width < img_gray.shape[0]
            )
        ]

        for (x, y) in valid_well_locs:
            cv2.circle(img_color2, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

        # ==== Debug Visualization ====
        if debug_well_detection:
            disp_img(img_color, img_color2, color=True)

        return valid_well_locs, img_color

    else:
        img_gray = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        print(f"Less than 10 points detected in this frame.")
        return None, img_gray