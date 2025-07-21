import os
import pdb
import shutil
import sys
from pathlib import Path
from argparse import ArgumentParser

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(str(PROJECT_ROOT))

data_dir = PROJECT_ROOT / 'data'
models_dir = PROJECT_ROOT   / 'models'
template_dir = PROJECT_ROOT / 'templates' / 'final_templates'

from preprocessing.organization import organize_image_data
from preprocessing.generate_frames import crop_stitched_images
from preprocessing.helpers.io_utils import read_json_file, create_outdir_name
from preprocessing.helpers.org_utils import create_well_translation_dict, flip_dict
from preprocessing.alignment import perform_frame_by_frame_alignment
from preprocessing.cleaning import remove_edge_frames


def organize_and_clean_data(
    project_dir,
    metadata_fpath: Path,
    perform_alignment: bool,
    clean_originals: bool,
    clean_post_alignment: bool,
    crop_stitched: bool,
    debug_alignment: bool,
    debug_cleaning: bool,
    debug_stitched_wkflow: bool
) -> list:

    data_fld_paths_dict = {}
    try:
        params_dict = read_json_file(metadata_fpath)
        data_fld_dict = params_dict["data_folders"]
    except KeyError:
        print(f"Metafile missing key 'data_folders'. Terminating...\n")
        sys.exit(1)

    # if dictionary containign data folder names is empty, exit
    if not data_fld_dict:
        print("Error: No data folders provided.")
        sys.exit(1)

    # Check if all values are either 'null' or ''
    all_null_or_empty = all(
        val.strip().lower() == "null" or val.strip() == ""
        for val in data_fld_dict.values()
    )

    if all_null_or_empty:
        print("Error: All data folder values are 'null' or empty.")
        sys.exit(1)

    # atleast one non-empty data folder name provided
    else:

        for key, val in data_fld_dict.items():
            # if dala folder name is not null or empty
            if val is not None or len(val) > 0:
                data_fld_path = os.path.join(os.path.dirname(metadata_fpath), val)

                # check if that exists
                if os.path.isdir(data_fld_path):
                    data_fld_paths_dict[key] = data_fld_path
                else:
                    print(f"User provided datafolder {data_fld_path} does no exist.")



    # check whether data is from 3-device holder or 96-well plate
    if 1 <= params_dict["num_wells"] <= 12:
        well_id_translation = create_well_translation_dict(params_dict)
        well_to_device_dict = flip_dict(params_dict)
    else:
        well_id_translation = None
        well_to_device_dict = None

    # provide path to organized folder
    uniq_fldname = create_outdir_name(params_dict)   # create unique foldername
    org_fld_path = project_dir / "data" / "organized" / uniq_fldname

    paths_to_indv_devices = []

    # organize data for each datafolder
    for session_num, data_fld_path in data_fld_paths_dict.items():

        # if session number starts with 0, shift all numbers by 1
        if int(sorted(list(data_fld_paths_dict.keys()))[0]) == 0:
            session_num = int(session_num) + 1

        # Organize image data
        res = organize_image_data(data_fld_path, org_fld_path, params_dict, session_num, well_to_device_dict, well_id_translation)

        paths_to_indv_devices.append(res[0])

    dev_fpaths_uniq = list(set([p for sublist in paths_to_indv_devices for p in sublist]))
    # Crop stitched image and save in different directory
    for dev_fpath in dev_fpaths_uniq:
        dev_fpath = Path(dev_fpath)

        stitched_path = dev_fpath / "original_stitched"

        if crop_stitched:
            # if stitched path exists, crop stitched imae into individual roi:
            if stitched_path.exists():
                # create path to save croped stitched image
                dev_dst_fpath = dev_fpath / "roi_frames"
                dev_metadata_path = stitched_path / 'metadata.json'
                dev_params_dict = read_json_file(dev_metadata_path)
                crop_percent = params_dict.get("crop_percent", 0.05)
                use_markers = bool(params_dict.get("use_markers", 0))
                crop_stitched_images(stitched_path, dev_dst_fpath, dev_params_dict,
                                     crop_percent=crop_percent, debug=False,
                                     use_markers=use_markers)
                shutil.rmtree(stitched_path)
            else:
                print(f"No stitched images found in this dataset.")



    return [dev_fpaths_uniq, crop_stitched, perform_alignment, clean_originals]

if __name__ == "__main__":
    # PROVIDE METADATA PATHS
    metadata_paths = [
        'data/C96005_U87_U251_cyt/96WP_metadata_cyt.json',
        'data/C96005_U87_U251_evos/C96005_metadata.json'
    ]

    # provide channel names for well detection and counting
    lf_dir = 'bright field'

    for metadata_path in metadata_paths:
        parser = ArgumentParser()

        parser.add_argument("--meta_data",
                            type=str,
                            default=f"{metadata_path}",
                            help="path to the metadata json file")

        args = parser.parse_args()
        params = read_json_file(args.meta_data)

        raw_metadata_path = Path(args.meta_data)

        # STEP 1: PERFORM PREPROCESSING AND ORGANIZATION
        if 'cytation' in params["microscope"]:
            perform_alignment = True
            clean_originals=False
        else:
            perform_alignment = False
            clean_originals =True

        res_step1 = organize_and_clean_data(
            PROJECT_ROOT,
            Path(metadata_path),
            crop_stitched=True,
            perform_alignment=False,
            clean_originals=False,
            clean_post_alignment=False,
            debug_alignment=False,
            debug_cleaning=False,
            debug_stitched_wkflow=False)