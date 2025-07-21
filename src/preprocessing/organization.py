import os
import re
import pdb
import sys
import shutil
import zipfile
from pathlib import Path
from tifffile import imread, imwrite
from typing import Union, Optional, Dict

import cv2
import numpy as np

parent_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(str(parent_dir))

from preprocessing.helpers.io_utils import write_json_file
from preprocessing.helpers.re_patterns import get_file_pattern


def too_long_path(path):
    path = Path(path).resolve()
    return f"\\\\?\\{path}"

def organize_data_evos(src_fld_path,
                       dest_fld_path,
                       session_number,
                       channel_id_to_channel_dict,
                       params_dict,
                       well_to_device_dict,
                       well_id_translation
                       ):
    well_ids = set()
    channels = set()
    num_images = 0
    paths_to_indv_devices = []

    pattern = get_file_pattern("evos", "raw")

    for file_path in src_fld_path.rglob("*.tif"):
        match = pattern.match(file_path.name)
        if not match:
            print(f"Filename did not match pattern: {file_path.name}")
            continue

        groups = match.groupdict()


        # if the images are not stitched images
        if groups.get("img_type") is not None and len(groups.get("img_type")) == 1:
            img_type = "original_frames"
        else:
            img_type = "original_stitched"

        num_images += 1
        well_id = groups.get("well_id")

        # Normalize to format like "D1", "E12", etc.
        well_match = re.match(r"([a-zA-Z])0*(\d+)", well_id)

        if well_match:
            letter = well_match.group(1).upper()
            number = int(well_match.group(2))  # automatically removes leading zeros
            well_id = f"{letter}{number}"
        else:
            print(f"Warning: unexpected well ID format '{well_id}'")
            well_id = well_id  # fallback in case of unexpected format

        frame_id = int(groups.get("frame_id"))

        # Resolve channel name
        try:
            channel_id = groups["channel_id"]
            channel = channel_id_to_channel_dict.get(channel_id, f"channel{channel_id}").lower()
            if 'bright' in channel:
                suffix = 'lf'
            else:
                suffix = 'lab'
        except KeyError:
            print("Key 'channel_id' not detected for image data from evos.")
            sys.exit(1)

        channels.add(channel)
        well_ids.add(well_id)

        # Handle device ID assignment
        device_id = params_dict.get("device_ID")
        # If user did not provide a device id but well_to_device_dict and well_id_translation are not None
        if (not device_id) and well_to_device_dict:
            device_id = well_to_device_dict.get(well_id, "C000")
            # if well_id_translation:
            #     well_id_new = well_id_translation.get(well_id, well_id)
            #     well_id = f"{well_id_new}_{well_id}"
        elif not device_id:
            device_id = "C000"

        # create  new filename removing unnecessary info
        # Construct output path
        session_folder = f"day{session_number}"
        # rename file removing channel name/number
        new_fname = f"{device_id}_{well_id}_f{frame_id}.tif"
        dest_img_path = dest_fld_path / device_id / f"{img_type}" / well_id / session_folder / channel / new_fname
        dest_img_path.parent.mkdir(parents=True, exist_ok=True)

        # Read the image with original bit depth (multi-channel or grayscale)
        try:
            img = imread(str(file_path), key=0)  # Automatically handles 8-bit, 16-bit, multi-channel
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            img = None

        # Proceed only if image was read successfully
        if img is not None:
            if img.dtype == np.uint8:
                # 8-bit image: save as-is
                imwrite(str(dest_img_path), img, photometric='minisblack')
            elif img.dtype == np.uint16:
                # Normalize to 16-bit full scale if it’s in 12-bit range
                img_max = img.max()
                if img_max <= 4095:
                    # Scale to 16-bit full range
                    img_norm = (img.astype(np.float32) / img_max * 65535).astype(np.uint16)
                else:
                    img_norm = img
                imwrite(str(dest_img_path), img_norm)
            else:
                print(f"Unexpected bit depth ({img.dtype}) in file: {file_path}")
        else:
            print(f"Failed to read image: {file_path}")

        print(f"Copied: {file_path} → {dest_img_path}")

    # Step 3: Write metadata
    for device_dir in dest_fld_path.iterdir():
        if not device_dir.is_dir():
            continue

        device_id = device_dir.name
        for img_type_dir in device_dir.iterdir():
            wells = [w.name for w in img_type_dir.iterdir() if w.is_dir()]

            # Make a deep copy to avoid modifying original dict
            device_params = params_dict.copy()
            device_params["well_IDs"] = wells
            device_params["device_ID"] = device_id

            if 1 <= device_params.get("num_wells", 0) <= 12:
                device_params["device_ID"] = device_id

            metadata_path = img_type_dir / "metadata.json"
            write_json_file(device_params, metadata_path)

        paths_to_indv_devices.append(device_dir)

    return [list(set(paths_to_indv_devices)), well_ids, channels, num_images, str(src_fld_path)]


def organize_data_cytation(src_fld_path,
                       dest_fld_path,
                       session_number,
                       params_dict,
                       well_to_device_dict,
                       well_id_translation):
    well_ids = set()
    channels = set()
    num_images = 0
    paths_to_indv_devices = []
    pattern = get_file_pattern("cytation", "raw")
    for file_path in src_fld_path.rglob("*.tif"):
        match = pattern.match(file_path.name)
        if not match:
            print(f"Filename did not match pattern: {file_path.name}")
            continue

        groups = match.groupdict()

        # if the images are not stitched images
        if groups.get("img_type_id") is not None and int(groups.get("img_type_id")) >= 0:
            img_type = "original_frames"
        else:
            img_type = "original_stitched"

        num_images += 1
        well_id = groups.get("well_id")
        frame_id = groups.get("frame_id")
        channel_id = groups.get('channel_id')

        # Resolve channel name
        try:
            channel = groups["channel"].lower()
            if "dapi" in channel:
                channel = "dapi"
                suffix = 'lab'
            elif "bright" in channel:
                channel = "bright field"
                suffix = 'lf'
            else:
                channel = channel
                suffix = 'lab'
        except KeyError:
            print("Key 'channel' not detected for image data from cytation.")
            sys.exit(1)

        channels.add(channel)
        well_ids.add(well_id)

        # Handle device ID assignment
        device_id = params_dict.get("device_ID")
        # If user did not provide a device id but well_to_device_dict and well_id_translation are not None
        if (not device_id) and well_to_device_dict:
            device_id = well_to_device_dict.get(well_id, "C000")
            params_dict["device_ID"] = device_id
            # if well_id_translation:
            #     well_id_new = well_id_translation.get(well_id, well_id)
            #     well_id = f"{well_id_new}_{well_id}"
        elif not device_id:
            device_id = "C000"

        # Construct output path
        session_folder = f"day{session_number}"

        # rename file removing channel name/number
        new_fname = f"{device_id}_{well_id}_f{frame_id}.tif"
        dest_img_path = dest_fld_path / device_id / f"{img_type}" / well_id / session_folder / channel / new_fname
        dest_img_path.parent.mkdir(parents=True, exist_ok=True)

        with open(too_long_path(file_path), 'rb') as src, open(dest_img_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)
        # shutil.copy(file_path, dest_img_path)

        print(f"Copied: {file_path} → {dest_img_path}")

    # Step 3: Write metadata
    for device_dir in dest_fld_path.iterdir():
        if not device_dir.is_dir():
            continue

        device_id = device_dir.name
        for img_type_dir in device_dir.iterdir():
            wells = [w.name for w in img_type_dir.iterdir() if w.is_dir()]

            # Make a deep copy to avoid modifying original dict
            device_params = params_dict.copy()
            device_params["well_IDs"] = wells
            if 1 <= device_params.get("num_wells", 0) <= 12:
                device_params["device_ID"] = device_id

            metadata_path = img_type_dir / "metadata.json"
            write_json_file(device_params, metadata_path)

        paths_to_indv_devices.append(device_dir)

    return [list(set(paths_to_indv_devices)), well_ids, channels, num_images, str(src_fld_path)]


def organize_image_data(
    src_fld_path: Union[str, Path],
    dest_fld_path: Union[str, Path],
    params_dict: Dict,
    session_number: int,
    well_to_device_dict: Optional[Dict] = None,
    well_id_translation: Optional[Dict] = None
):
    src_fld_path = Path(src_fld_path)
    dest_fld_path = Path(dest_fld_path)

    if not src_fld_path.exists():
        print(f"Source path does not exist: {src_fld_path}")
        return None

    # Unzip if zipped
    if zipfile.is_zipfile(src_fld_path):
        unzip_dir = src_fld_path.with_suffix("")
        with zipfile.ZipFile(src_fld_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        try:
            src_fld_path.unlink()
            print(f"Deleted zip file: {src_fld_path}")
        except Exception as e:
            print(f"Could not delete zip file: {e}")
        src_fld_path = unzip_dir
    else:
        print(f"Input is not a zip file: {src_fld_path}")

    system = params_dict.get("microscope", "").lower()
    channel_id_to_channel_dict = params_dict.get("channels_dict", {})

    # Compile filename pattern
    if system == "cytation":
        return organize_data_cytation(src_fld_path,
                                dest_fld_path,
                                session_number,
                                params_dict,
                                well_to_device_dict,
                                well_id_translation)
    elif system == "evos":
        return organize_data_evos(src_fld_path,
                       dest_fld_path,
                       session_number,
                       channel_id_to_channel_dict,
                       params_dict,
                       well_to_device_dict,
                       well_id_translation
                       )

