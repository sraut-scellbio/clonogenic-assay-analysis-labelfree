import pdb
from pathlib import Path

from natsort import natsorted
from tifffile import TiffFile

from src.preprocessing.helpers.crop_utils import detect_well_roi, crop_and_save, get_crop_params_from_first_bf, \
    save_frames_from_cropped_stitched_image
from src.preprocessing.helpers.io_utils import write_json_file, read_json_file


def crop_stitched_images(dev_raw_stitched_img_folder, output_folder, params_dict, crop_percent=0.05, debug=False, use_markers=False):
    dev_raw_stitched_img_folder = Path(dev_raw_stitched_img_folder)
    output_folder = Path(output_folder)
    params_dict_dev = params_dict.copy()
    params_dict_dev["crop_percent"] = crop_percent
    params_dict_dev["crop_area"] = {}
    params_dict_dev["crop_coordinates"] = {}


    for well_folder in dev_raw_stitched_img_folder.iterdir():
        if not well_folder.is_dir():
            continue  # Skip if it's not a directory
        well_folder_name = well_folder.name
        params_dict_dev["crop_area"][well_folder_name] = {}
        params_dict_dev["crop_coordinates"][well_folder_name] = {}
        if not well_folder.is_dir():
            continue

        crop_params = None

        # 1. get max_height and width cropping params for first bf image
        for day_folder in natsorted(well_folder.iterdir()):
            brightfield_dir = None
            for subfolder in day_folder.iterdir():
                if 'bright' in subfolder.name.lower():
                    brightfield_dir = subfolder
                    break

            if not brightfield_dir:
                print(f"No brightfield folder in {day_folder}")
                continue

            brightfield_image = next(brightfield_dir.glob("*.tif"))

            with TiffFile(brightfield_image) as tif:
                print(f"Number of pages: {len(tif.pages)} (reading bf image to get crop params)")
                brightfield = tif.pages[0].asarray()
            crop_params = get_crop_params_from_first_bf(brightfield, crop_percent=crop_percent)
            break

        '''
        Note:
        1. We use the same cropping parameters i.e. height and width of the crop box across different
           days to maek sure the size of teh cropped image remains same (multiple of 1224 and 904).
        2. We use the exact same crop coordinates (x1, y1, x2, y2) for each channel in a day/session folder.
        3. Crop params remain same across days while crop coordinates remain same across channels but the center may vary.
        '''
        # 2. get actual crop coordinates based on well center and crop parameters calculated from day1 bf images
        for day_folder in well_folder.iterdir():
            day_folder_name = day_folder.name
            if not day_folder.is_dir():
                continue

            brightfield_dir = None
            for subfolder in day_folder.iterdir():
                if 'bright' in subfolder.name.lower():
                    brightfield_dir = subfolder
                    break

            if not brightfield_dir:
                print(f"No brightfield folder in {day_folder}")
                continue

            brightfield_image = next(brightfield_dir.glob("*.tif"))
            with TiffFile(brightfield_image) as tif:
                print(f"Number of pages: {len(tif.pages)} (reading for crop dayfolder: {day_folder_name})")
                brightfield = tif.pages[0].asarray()

            """
            1. roi_box_coords: bbox coords for region ot interest for extracting frames
            2. reference_marker_coords: bbox coords for reference point(either top left well corner or top left marker within ROI dependng on whether 
            use_markers=True)
            3. well_rect_coords: bbox defining well area
            """
            roi_box_coords, reference_marker_coords, well_rect_coords = detect_well_roi(brightfield, crop_params, params_dict_dev, debug=debug,
                                                                                        title=f"{well_folder_name}"
                                                                                              f" {day_folder_name}", use_markers=use_markers)

            x1, y1, x2, y2 = roi_box_coords
            crop_area = (x2 - x1) * (y2 - y1)

            params_dict_dev["crop_area"][well_folder_name][day_folder_name] = crop_area
            params_dict_dev["crop_coordinates"][well_folder_name][day_folder_name] = [x1, y1, x2, y2]

            # Use the same crop region for other channels
            for channel_folder in day_folder.iterdir():
                if not channel_folder.is_dir():
                    continue

                for tif_file in channel_folder.glob("*.tif"):
                    rel_path = tif_file.relative_to(dev_raw_stitched_img_folder)
                    save_path = output_folder / rel_path
                    save_path_stitched = save_path.parents[0] / "stitched" / tif_file.name

                    # get crop for entire well
                    orig_img, _ = crop_and_save(tif_file, None, save_path_stitched, save=False)
                    cropped_well, _ = crop_and_save(tif_file, well_rect_coords, save_path_stitched, save=False)

                    # # get center crop
                    cropped_roi, orig = crop_and_save(tif_file, roi_box_coords, save_path_stitched, save=False, verbose=True)
                    save_path_frames = save_path.parents[0]
                    save_frames_from_cropped_stitched_image(cropped_roi, orig_img, save_path_frames, float(params_dict_dev["resolution"]),
                                                            tif_file.name,
                                                            roi_box_coords, reference_marker_coords, params_dict)


    # save device metadata
    metadata_save_path = Path(output_folder) / "metadata.json"
    write_json_file(params_dict_dev, metadata_save_path)

