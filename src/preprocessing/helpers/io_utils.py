import os
import re
import sys
import pdb
import json
import zipfile
import natsort
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

def disp_img(*imgs, color=False, title=''):
    if len(imgs) == 1:
        # Display one image
        if not color:
            plt.imshow(imgs[0], cmap='gray')
        else:
            plt.imshow(imgs[0])
        plt.axis('off')
        plt.title(title)
        plt.show()
    elif len(imgs) == 2:
        # Display two images side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, img in zip(axes, imgs):
            if not color:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.title(title)
        plt.show()
    else:
        raise ValueError("This function only supports displaying one or two images.")

def save_img(*imgs, fname, out_dir):
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    if len(imgs) == 1:
        # Save one image
        plt.imshow(imgs[0], cmap='gray')
        plt.axis('off')
        file_path = os.path.join(out_dir, fname)
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    elif len(imgs) == 2:
        # Save two images side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        for ax, img in zip(axes, imgs):
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        file_path = os.path.join(out_dir, fname)
        plt.tight_layout()
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        raise ValueError("This function only supports saving one or two images.")


def read_json_file(json_file_path):
    # Check if the file exists
    if not os.path.isfile(json_file_path):
        raise FileNotFoundError(f"The file '{json_file_path}' does not exist.")

    # Read and parse the JSON file
    with open(json_file_path, "r") as json_file:
        data = json.load(json_file)

    return data


def write_json_file(params_dict, save_path):
    # Ensure the directory for the save path exists
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Write the dictionary to the JSON file
    with open(save_path, "w") as json_file:
        json.dump(params_dict, json_file, indent=4)


def get_images_as_array(fpath_lf, fpath_fluo, img_size=None):
    n_c, n_r = tuple(img_size)
    images_lf, fnames_lf = [], []
    images_fluo, fnames_fluo = [], []

    # Process first folder
    fnames_lf = os.listdir(fpath_lf)
    sorted_fnames_lf = natsort.natsorted(fnames_lf)
    for fname in sorted_fnames_lf:
        file_path = os.path.join(fpath_lf, fname)
        try:
            with Image.open(file_path) as img:
                images_lf.append(np.array(img))
                fnames_lf.append(fname)
        except IOError:
            print(f"Skipping non-image file: {file_path}")

    if fpath_fluo is not None:
        # Process second folder
        fnames_fluo = os.listdir(fpath_fluo)
        sorted_fnames_fluo = natsort.natsorted(fnames_fluo)
        for fname in sorted_fnames_fluo:
            file_path = os.path.join(fpath_fluo, fname)
            try:
                with Image.open(file_path) as img:
                    images_fluo.append(np.array(img))
                    fnames_fluo.append(fname)
            except IOError:
                print(f"Skipping non-image file: {file_path}")

        if fnames_lf != fnames_fluo:
            # Convert lists to sets for easy comparison
            set1 = set(fnames_lf)
            set2 = set(fnames_fluo)

            # Files in fnames_lf but not in fnames_fluo
            missing_in_fnames2 = set1 - set2
            # Files in fnames_fluo but not in fnames_lf
            missing_in_fnames1 = set2 - set1

            '''
            Check if it is a good idea to delete or create blank frames for missing frames
            '''
            # if missing_in_fnames2:
            #     print("Files present in fluo but missing in lf:")
            #     for filename in missing_in_fnames2:
            #         print(" -", filename)
            #         path_in_fpath_lf = os.path.join(fpath_lf, filename)
            #         try:
            #             os.remove(path_in_fpath_lf)
            #             print(f"Deleted '{path_in_fpath_lf}' because it is missing in {fpath_fluo}.")
            #         except Exception as e:
            #             print(f"Failed to delete '{path_in_fpath_lf}': {e}")
            #
            # if missing_in_fnames1:
            #     print("Files present in lf but missing in fluo:")
            #     for filename in missing_in_fnames1:
            #         print(" -", filename)
            #         path_in_fpath_fluo = os.path.join(fpath_fluo, filename)
            #         try:
            #             os.remove(path_in_fpath_fluo)
            #             print(f"Deleted '{path_in_fpath_fluo}' because it is missing in {fpath_lf}.")
            #         except Exception as e:
            #             print(f"Failed to delete '{path_in_fpath_fluo}': {e}")

            return np.array(images_lf), sorted_fnames_lf, np.array(images_fluo), sorted_fnames_fluo

    else:
        return np.array(images_lf), sorted_fnames_lf



def get_all_img_stack_paths_as_dicts(well_path):
    paths_dict = {}

    day_flds = os.listdir(well_path)
    for day_fld in day_flds:
        paths_dict[day_fld] = {}
        channel_flds = os.listdir(os.path.join(well_path, day_fld))
        for channel_fld in channel_flds:
            img_stack_path = os.path.join(os.path.join(well_path, day_fld, channel_fld))
            paths_dict[day_fld][channel_fld] = img_stack_path

    return paths_dict


def extract_files_and_get_wells_and_channels_list(zip_fld_path, system='cytation'):

    if system == 'cytation':
        well_ids = set()
        channels = set()
        num_images = 0
        pattern = re.compile(r"^([^_]+)_\d+_(\d+)_\d+_([a-zA-Z\s]+)_(\d{3})\.tif$")
        is_zip_file = True

        # Step 1: Check if folder is a zipped file and unzip it if necessary
        if zipfile.is_zipfile(zip_fld_path):
            unzip_dir = zip_fld_path.replace(".zip", "")
            with zipfile.ZipFile(zip_fld_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            unzipped_fld_path = unzip_dir

            # Remove the original .zip file after extraction
            try:
                os.remove(zip_fld_path)
                print(f"Deleted zip file: {zip_fld_path}")
            except OSError as e:
                print(f"Error deleting zip file {zip_fld_path}: {e}")
        else:
            print(f"\nFolder at {zip_fld_path} is not a .zip folder.")
            is_zip_file = False
            unzipped_fld_path = zip_fld_path

        # Step 2: Iterate through extracted folder and gather well/channel info
        for root, dirs, files in os.walk(unzipped_fld_path):
            for file in files:
                if file.endswith(".tif"):
                    num_images += 1
                    match = pattern.match(file)
                    if match:
                        well_id, third_number, channel, _ = match.groups()
                        well_ids.add(well_id)
                        channels.add(channel)

        print(f"Folder at {unzipped_fld_path} contains data for:\n Wells: {well_ids}\nChannels: {channels}")

        return well_ids, channels, num_images, unzipped_fld_path

    if system == 'evos':
        well_ids = set()
        channel_ids = set()
        num_images = 0
        pattern = r'_(?P<img_type>[A-Z]+)_.*_(?P<wellID>[A-Z]\d+)(?P<frameID>f\d+)(?P<channelID>d\d+)'
        is_zip_file = True

        # Step 1: Check if folder is a zipped file and unzip it if necessary
        if zipfile.is_zipfile(zip_fld_path):
            unzip_dir = zip_fld_path.replace(".zip", "")
            with zipfile.ZipFile(zip_fld_path, 'r') as zip_ref:
                zip_ref.extractall(unzip_dir)
            unzipped_fld_path = unzip_dir

            # Remove the original .zip file after extraction
            try:
                os.remove(zip_fld_path)
                print(f"Deleted zip file: {zip_fld_path}")
            except OSError as e:
                print(f"Error deleting zip file {zip_fld_path}: {e}")
        else:
            print(f"\nFolder at {zip_fld_path} is not a .zip folder.")
            is_zip_file = False
            unzipped_fld_path = zip_fld_path

        # Step 2: Iterate through extracted folder and gather well/channel info
        for root, dirs, files in os.walk(unzipped_fld_path):
            for file in files:
                if file.lower().endswith(".tif"):
                    num_images += 1
                    match = re.search(pattern, file)
                    if match:
                        img_type = match.group('img_type')
                        well_id = match.group('wellID')
                        channel_id = int(match.group('channelID')[1:])  # remove 'd' and convert to int
                        well_ids.add(well_id)
                        channel_ids.add(channel_id)

        print(f"Folder at {unzipped_fld_path} contains data for:\n Wells: {well_ids}\nChannels: {channel_ids}")

        return well_ids, channel_ids, num_images, unzipped_fld_path


def create_outdir_name(params_dict, well_id=None):
    uname = params_dict.get("user_name", "user")
    exp_date = params_dict.get("exp_date", "mm_dd_yy")
    keyword = params_dict.get("keyword", "keywrd")
    cell_type = params_dict.get("cell_line", "u87")
    microscope = params_dict.get("microscope", "cytation")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        devices = "_".join([key for key in params_dict["device_to_well_dict"].keys()])
    except KeyError:
        print("Warning KeyError: 'device_to_well_dict' not found in params_dict. Trying key 'device_ID' instead for results fname.")
        try:
            devices = params_dict["device_ID"]
            print("Used 'device_id' from metadata file for creating results fname.")
        except KeyError:
            print("Warning KeyError: 'device_ID' not found in params_dict for creating filename. Using "" in place.")
            devices = ""

    if well_id is not None:
        fld_name = f"{exp_date}_{devices}_{microscope}_{uname}_{keyword}_{well_id}_{cell_type}_{timestamp}"
    else:
        fld_name = f"{exp_date}_{devices}_{microscope}_{uname}_{keyword}_{cell_type}_{timestamp}"
    return fld_name