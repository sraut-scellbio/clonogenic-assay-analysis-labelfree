import os
import stat
from pathlib import Path
from argparse import ArgumentParser

from natsort import natsorted

from filetypes import image_types
from helpers.re_patterns import get_file_pattern
from helpers.io_utils import read_json_file
from helpers.cleaning_utls import is_almost_black, is_edge_frame


# empty frames from label-free channel
def get_invalid_frame_names_lf(lf_folder, debug_invalid_frames=False):
    invalid_frame_names = []

    # Step 1: Identify black images in 'day1/images_lf'
    if os.path.exists(lf_folder):
        for file_name in os.listdir(lf_folder):
            if file_name.lower().endswith('.tif'):
                file_path = os.path.join(lf_folder, file_name)
                if is_almost_black(file_path, debug=debug_invalid_frames) or is_edge_frame(file_path, debug=debug_invalid_frames):
                    invalid_frame_names.append(file_name)


    return invalid_frame_names


def remove_frames(frame_id_list, folder_path, pattern):

    for filepath in folder_path.iterdir():
        if filepath.is_file() and filepath.suffix in image_types.VALID_IMAGE_EXTENSIONS:
            frame_id = pattern.match(filepath.name).groupdict().get("frame_id")
            if frame_id in frame_id_list:
                try:
                    os.remove(filepath)
                except PermissionError:
                    print(f"File {filepath.name} removal failed. Changing permissions and attempting removal.")
                    try:
                        os.chmod(filepath, stat.S_IWRITE)  # Make it writable
                        os.remove(filepath)
                        print(f"Removed image {filepath} as empty or edge frame.")
                    except Exception as e:
                        print(f"Still couldn't delete {filepath}: {e}")



def remove_edge_frames(well_fpath, params, debug_invalid_frames=False):

    well_fpath = Path(well_fpath)
    well_ID = well_fpath.name
    device_ID = well_fpath.parents[1].name

    # Find the brightfield folder in day1 using 'bright' keyword
    # Automtically find the name of the start date folder
    exp_start_fldname = natsorted(os.listdir(well_fpath))[0]
    start_spath = well_fpath / exp_start_fldname

    try:
        bf_folder_candidates = [f for f in start_spath.iterdir() if f.is_dir() and "bright" in f.name.lower()]
    except Exception as e:
        print(f"Error {e}. Skipping edge frame removal for well {well_ID}.")
        return
    if not bf_folder_candidates:
        raise ValueError(f"No brightfield folder found in 'day1' under {well_fpath}")

    bf_channel = bf_folder_candidates[0].name  # use the matched brightfield folder name

    # --- Get all day folders ---
    day_folders = sorted([d for d in well_fpath.iterdir() if d.is_dir()])

    if not day_folders:
        print(f"No 'day*' folders found in {well_fpath}")
        return

    # --- Get reference bright field from day1 ---
    start_spath = next((d for d in day_folders if d.name == exp_start_fldname), None)
    if not start_spath:
        raise ValueError(f"{exp_start_fldname} folder not found in {well_fpath}")

    day1_bf_path = start_spath / bf_channel

    # get name of files with emoty or edge frames
    # pdb.set_trace()
    fnames_to_remove = get_invalid_frame_names_lf(day1_bf_path, debug_invalid_frames=debug_invalid_frames)

    # get frame number from frame names
    pattern = get_file_pattern(params["microscope"], "orig")
    frame_ids_to_remove = [pattern.match(frame_name).groupdict().get("frame_id") for frame_name in fnames_to_remove]

    for day_fldpath in day_folders:
        for channel_fldpath in day_fldpath.iterdir():
            if channel_fldpath.is_dir():
                remove_frames(frame_ids_to_remove, channel_fldpath, pattern)
            else:
                print(f"{channel_fldpath.name} not a channel folder. Skipping invalid frame removal.")



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


        args = parser.parse_args()
        file_paths = vars(args)
        params = read_json_file(args.meta_data)

        for well_ID in params["well_IDs"]:
            well_fpath = os.path.join(os.path.dirname(args.meta_data), well_ID)
            remove_edge_frames(well_fpath, params)