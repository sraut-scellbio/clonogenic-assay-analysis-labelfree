import os
import pdb
import re
import sys
from pathlib import Path
from argparse import ArgumentParser

from analysis.result_generation_helpers import create_array_heatmap

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(str(PROJECT_ROOT))


from analysis import indv_analysis, tracking_analysis
from preprocessing.helpers.io_utils import read_json_file


if __name__ == "__main__":
    metadata_paths = [
        'data/organized/06_30_25_CT2_evos_SCB_bub-test_None_2025-07-07_13-53-27/CT2/results/roi_frames/metadata.json'
    ]


    for metadata_path in metadata_paths:
        parser = ArgumentParser()

        parser.add_argument("--meta_data",
                            type=str,
                            default=f"{metadata_path}",
                            help="path to the metadata json file")

        args = parser.parse_args()
        params = read_json_file(args.meta_data)
        metadata_path = Path(args.meta_data)
        device_dir = metadata_path.parents[2]
        params["device_ID"] = device_dir.name
        frame_type = metadata_path.parent.name
        results_dir = device_dir / 'results' / frame_type
        # wells = ["A4", "A5", "A6", "D1", "D5", "D6", "A9", "A10", "A11", "D9"]
        # wells = ["A3", "A5", "A6", "D1", "D2", "D3", "A7", "A8", "A9", "D8"]
        wells = ["E6", "E7"]
        for well_fldpath in metadata_path.parent.iterdir():
            if well_fldpath.is_dir() and well_fldpath.name in wells:
                indv_analysis.save_per_day_summary(well_fldpath, params)
                tracking_analysis.process_well_predictions(well_fldpath)


        # # create clonogeninc index heatmap
        # summary_fld_name = metadata_path.parts[-5]
        # results_summary_path = Path(f"results_summary/{summary_fld_name}")
        #
        # # create heatmap for single cell clones
        # pattern = re.compile(rf"{re.escape(device_dir.name)}?_*([A-H])(\d{{1,2}})_cidx\.npy$")
        # create_array_heatmap(results_summary_path, device_dir.name, pattern, cmap="coolwarm")
        #
        # # create heatmap for all clones
        # pattern = re.compile(rf"{re.escape(device_dir.name)}?_*([A-H])(\d{{1,2}})_cidx_allwells\.npy$")
        # create_array_heatmap(results_summary_path, device_dir.name, pattern, cmap="coolwarm", suffix="cindex_heatmap_allwells")