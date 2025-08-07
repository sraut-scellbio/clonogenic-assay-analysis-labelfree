import os
import pdb
import sys
import time
import cv2
import joblib
import numpy as np
from os import makedirs
from pathlib import Path
from typing import Union
from natsort import natsorted
from argparse import ArgumentParser

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(str(PROJECT_ROOT))

data_dir = PROJECT_ROOT / 'data'
models_dir = PROJECT_ROOT   / 'models'
templates_dir = PROJECT_ROOT / 'templates' / 'final_templates'

from preprocessing.coeffs import film_params
from preprocessing.filetypes import image_types
from processing.load_labelfree import load_cellpose_model
from processing.well_identification import get_well_locations
from preprocessing.helpers.io_utils import read_json_file, write_json_file
from processing.intensity_calculation import get_avg_single_cell_intensity
from processing.count_cells import save_cell_counts_combined, save_cell_counts_labelfree


def save_raw_counts(well_fldpath: Union[Path, str],
         params_dict,
         template,
         out_dir=None,
         debug_cell_count=False,
         debug_well_detection=False,
         model_dir=None
)->Union [Path, str]:

    well_id = well_fldpath.name
    film_version = params_dict.get("template", "v1").lower()
    well_width_microns = film_params.well_width_microns.get(film_version, 51)  # microns
    bridge_width_microns = film_params.bridge_width_microns.get(film_version, 51)  # microns
    microns_per_pixel = float(params_dict.get("resolution", 0.645))  # default resolution: 1.0 μm/pixel
    well_width_pixels = well_width_microns / microns_per_pixel
    bridge_width_pixels = bridge_width_microns / microns_per_pixel
    cell_line = params_dict.get("cell_line", None)
    if cell_line is not None:
        cell_line = cell_line.lower()
    else:
        cell_line = params_dict.get("cell_lines", {}).get(well_id, None)
        if not cell_line:
            print(f"ERROR: 'cell_line' not provided and no entry found for well '{well_id}' in 'cell_lines'.")
            sys.exit(1)
        else:
            cell_line = cell_line.lower()
            print(f"Cell line for well {well_id} is {cell_line}.\n")
    model_path = os.path.join(model_dir, cell_line.lower(), "random_forests", 'model.pkl')

    # count cells for each day folder in the well folder
    start_sname = natsorted(os.listdir(well_fldpath))[0]   # get name of start day folder
    params_dict["start_sname"] = start_sname

    # update metadata with session start folder name
    res_metadata_path = os.path.join(out_dir, 'metadata.json')
    if not os.path.exists(res_metadata_path):
        write_json_file(params_dict, os.path.join(out_dir, 'metadata.json'))

    for day_fldpath in well_fldpath.iterdir():
        if day_fldpath.is_dir():
            curr_sname = day_fldpath.name
            lf_dir = day_fldpath / params_dict["lf_dir"]

            if not lf_dir.exists():
                print(f"{lf_dir.name} does not exist. Cell can not be counted for day folder {day_fldpath.name}.")
                continue

            # STEP 1: For termination day calculate single cell intensity
            fluo_channel = params_dict.get("fluo_dir", None)
            fluo_dir = day_fldpath / fluo_channel if fluo_channel is not None else None

            # if fluroscent folder does not exist, set intensity to None
            if fluo_dir is not None and fluo_dir.exists():
                model = None
                # load cell classification model and templates
                try:
                    # Load cell classification model
                    model = joblib.load(model_path)
                    print(f"Loaded model from {model_path}")
                except FileNotFoundError:
                    print(f"Model file not found at {model_path}. Setting model to None.")
                except Exception as e:
                    print(f"Error loading model: {e}. Setting model to None.")

                avg_sc_intensity = get_avg_single_cell_intensity(
                    lf_dir,
                    fluo_dir,
                    template,
                    microns_per_pixel,
                    cell_line,
                    well_width_pixels,
                    bridge_width_pixels,
                    model,
                    debug=debug_cell_count
                )
                cellpose_model=None

            else:
                print(f"Fluorescent directory not found. Cells will be counted using label free channel for {curr_sname}.")
                model = None
                avg_sc_intensity = None
                cellpose_model = load_cellpose_model(models_dir)

            # create output directory
            session_out_dir = out_dir / well_id / 'raw_counts' / curr_sname
            os.makedirs(session_out_dir, exist_ok=True)

            # detect wells and cells
            for i, lf_img_path in enumerate(lf_dir.iterdir()):

                if lf_img_path.is_file() and lf_img_path.suffix in image_types.VALID_IMAGE_EXTENSIONS:
                    img_name = lf_img_path.name

                    # STEP 2: Detect wells
                    well_locs, detection_results = get_well_locations(lf_img_path, well_width_pixels, bridge_width_pixels,
                                                                      template, microns_per_pixel,
                                                                   debug_well_detection=debug_well_detection)


                    if well_locs:
                        fluo_img_path = fluo_dir / img_name if avg_sc_intensity else None

                        if fluo_img_path is not None:
                            save_cell_counts_combined(fluo_img_path, lf_img_path, well_locs, well_width_pixels, cell_line, microns_per_pixel,
                                                  avg_sc_intensity=avg_sc_intensity,
                                                  model=model, out_dir=session_out_dir, debug=debug_cell_count, save_cropped_wells=True,
                                                  write_results=True)
                        else:
                            flow_threshold = 0.75
                            cellprob_threshold = 0.0
                            tile_norm_blocksize = 0
                            save_cell_counts_labelfree(cellpose_model,
                                                        flow_threshold,
                                                        cellprob_threshold,
                                                        tile_norm_blocksize,
                                                        lf_img_path,
                                                        well_locs,
                                                        cell_line,
                                                        microns_per_pixel,
                                                        well_width_pixels,
                                                        out_dir=session_out_dir,
                                                        write_results=True,
                                                        save_masks_for_training=True,
                                                        save_flows=True)
                    # no wells detected
                    else:
                        print(f"No wells detected for frame {img_name}.")
                        log_dir = session_out_dir / 'undetected_wells'
                        os,makedirs(log_dir, exist_ok=True)
                        save_path = log_dir / f'{img_name}'
                        cv2.imwrite(save_path, detection_results)
                        continue

                # invalid file
                else:
                    print(f"Skipping file {lf_img_path.name}. Invalid file. Check extension.")

    return out_dir


if __name__ == "__main__":
    metadata_paths = [
        'data/organized/C367_U87_evos/roi_frames/metadata.json',
        'data/organized/C373_U251_evos/roi_frames/metadata.json'
    ]

    for metadata_path in metadata_paths:
        parser = ArgumentParser()

        parser.add_argument("--meta_data",
                            type=str,
                            default=f"{metadata_path}",
                            help="path to the metadata json file")


        args = parser.parse_args()
        params = read_json_file(args.meta_data)

        params["lf_dir"] = "bright field"
        # params["fluo_dir"] = "dapi"
        metadata_path = Path(args.meta_data)
        device_dir = metadata_path.parents[1]
        frame_type = metadata_path.parent.name
        results_dir = device_dir / 'results' / frame_type
        os.makedirs(str(results_dir), exist_ok=True)

        # load template for device
        film_version = params.get("template", "v1").lower()
        well_template_path = os.path.join(templates_dir, "well", "test", f"template_{film_version}.npy")
        template = np.load(well_template_path).astype(np.uint8)
        print(f"Well templates loaded from {well_template_path}. Template shape : {template.shape}.")

        for well_fldpath in metadata_path.parent.iterdir():
            if well_fldpath.is_dir():
                print(f"\nProcessing: {well_fldpath.name}")
                start_time = time.time()
                save_raw_counts(well_fldpath, params,
                                template=template,
                                out_dir = results_dir,
                                debug_cell_count=False,
                                debug_well_detection=False,
                                model_dir=models_dir)

                elapsed = time.time() - start_time
                print(f"✅ Finished {well_fldpath.name} in {elapsed:.2f} seconds")
