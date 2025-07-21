import os
import pdb
from pathlib import Path
from argparse import ArgumentParser

from natsort import natsorted

from helpers.io_utils import write_json_file, read_json_file
from helpers.alignment_utils import calculate_shift_montage
from helpers.alignment_utils import align_and_crop_frames, get_montage_from_frames

def perform_frame_by_frame_alignment(well_fpath, dst_well_fpath, params, align_w_montage=True, debug_alignment=False):
    well_fpath = Path(well_fpath)
    well_ID = well_fpath.name
    device_ID = well_fpath.parents[1].name
    n_grid_rows, n_grid_cols = params["montage_grid"]
    overlap_x, overlap_y = params["frame_overlaps"]
    microns_per_pixel = params["resolution"]

    # Automtically find the name of the start date folder
    start_sname = natsorted(os.listdir(well_fpath))[0]
    start_sfpath = well_fpath / start_sname
    try:
        bf_folder_candidates = [f for f in start_sfpath.iterdir() if f.is_dir() and "bright" in f.name.lower()]
    except Exception as e:
        print(f"Error {e}. Skipping well {well_ID} for alignment.")
        return

    if not bf_folder_candidates:
        raise ValueError(f"No brightfield folder found in 'day1' under {well_fpath}")

    bf_channel = bf_folder_candidates[0].name  # use the matched brightfield folder name

    # --- Get all day folders ---
    day_folders = natsorted([d for d in well_fpath.iterdir() if d.is_dir() and d.name.startswith("day")])
    if not day_folders:
        print(f"No 'day*' folders found in {well_fpath}")
        return

    # --- Get reference bright field from day1 ---
    start_sfpath = next((d for d in day_folders if d.name == start_sname), None)
    if not start_sfpath:
        raise ValueError(f"{start_sname} folder not found in {well_fpath}")

    start_s_bf_path = start_sfpath / bf_channel
    start_s_bf_montage = get_montage_from_frames(start_s_bf_path, start_sname, bf_channel, (n_grid_rows, n_grid_cols), params, None, save=False)

    # --- Align all other days to day1 ---
    for day_folder in day_folders:
        if day_folder.name == start_sname:
            align_and_crop_frames(None, well_fpath, start_sname, dst_well_fpath, day_folder.name, None, None, None, None, overlap_x, overlap_y,
                                  microns_per_pixel,
                                  debug=debug_alignment)
        else:
            day_name = day_folder.name
            day_bf_path = day_folder / bf_channel

            day_bf_montage = get_montage_from_frames(day_bf_path, day_name, bf_channel, (n_grid_rows, n_grid_cols), params, None, save=False)

            shift_dict = calculate_shift_montage(start_s_bf_montage, day_bf_montage, debug=debug_alignment)
            print(f"[{day_name}] shift from stitched image:\n{shift_dict}")
            num_tr_grid_rows, num_tr_grid_cols, row_start, col_start = n_grid_rows, n_grid_cols, 0, 0

            align_and_crop_frames(shift_dict, well_fpath, start_sname, dst_well_fpath, day_folder.name, num_tr_grid_rows, num_tr_grid_cols,
                                                                       row_start, col_start,
                                  overlap_x, overlap_y, microns_per_pixel, debug=debug_alignment)



if __name__ == "__main__":
    metadata_paths = [
        'data/organized/03_31_25_cytation_SCB_4W-PDMS-markers_CPDMS1_multiday_U87_2025-04-29_15-12-11/CPDMS1/original_frames/metadata.json'
    ]

    for metadata_path in metadata_paths:
        parser = ArgumentParser()

        parser.add_argument("--meta_data",
                            type=str,
                            default=f"{metadata_path}",
                            help="path to the metadata json file")

        # parser.add_argument("--well_fpath",
        #                     type=str,
        #                     default=f"data/organized/02_06_25_SCB__C290_C292_C293_multiday_U87_2025-02-27_10-16-57/C292/wellA_B5",
        #                     help="path to day1 images")

        args = parser.parse_args()
        file_paths = vars(args)
        params = read_json_file(args.meta_data)

        for well_ID in params["well_IDs"]:
            well_fpath = os.path.join(os.path.dirname(args.meta_data), well_ID)
            perform_frame_by_frame_alignment(well_fpath, params)