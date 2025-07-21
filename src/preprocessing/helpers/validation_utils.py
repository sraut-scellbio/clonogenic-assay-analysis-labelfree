import os
import pdb
import shutil
from .io_utils import read_json_file, write_json_file

# returns path to unzipped folders and the common well_names/channels
def delete_missing_wells(day1_info, dayn_info):


    well_paths_d1, _, channels_d1, _, _ = day1_info
    well_paths_dn, _, channels_dn, _, _ = dayn_info

    # Compare channels and print missing ones
    missing_in_dayn = set(channels_d1) - set(channels_dn)
    missing_in_day1 = set(channels_dn) - set(channels_d1)

    if missing_in_dayn:
        print(f"Channels missing in DayN: {missing_in_dayn}")
    if missing_in_day1:
        print(f"Channels missing in Day1: {missing_in_day1}")

    # Compare well paths and delete missing folders
    wells_d1_set = set(well_paths_d1)
    wells_dn_set = set(well_paths_dn)

    path_extra_wells_in_dn =  wells_dn_set - wells_d1_set
    path_extra_wells_in_d1 =  wells_d1_set - wells_dn_set

    if path_extra_wells_in_d1:
        for path_extra_well in path_extra_wells_in_d1:
            print(f"Well {os.path.basename(path_extra_well)} is missing in day-n data.")
        print(f"These wells will be removed from processing.")
    if path_extra_wells_in_dn:
        for path_extra_well in path_extra_wells_in_dn:
            print(f"Well {os.path.basename(path_extra_well)} is missing in day-1 data.")
        print(f"These wells will be removed from processing.")

    path_extra_wells = path_extra_wells_in_dn | path_extra_wells_in_dn


    for path_extra_well in path_extra_wells:

        '''
        Check if this is worth doing.
        '''
        # remove well_ID that is about to be deleted from metadata
        well_basedir = os.path.dirname(path_extra_well)
        metadata_path = os.path.join(well_basedir, "metadata.json")
        params_dict = read_json_file(metadata_path)
        extra_wellid = os.path.basename(path_extra_wells)
        params_dict["well_IDs"].remove(extra_wellid)
        params_dict["missing_data"] = params_dict["missing_data"] + f" Yes-{extra_wellid}"
        write_json_file(params_dict, metadata_path)

        # Delete missing well folder
        print(f"Deleting {path_extra_well} (missing in DayN)")
        shutil.rmtree(path_extra_well, ignore_errors=True)

    path_common_wells = wells_dn_set & wells_d1_set
    return list(path_common_wells)


# checks whether the metadata provided by the user corresponds to the images presen in the file
def validate_wells_and_channels(set_wells_found, set_channels_found, params_dict):

    wells_in_meta = []
    for device_id, wells_list in params_dict["device_to_well_dict"].items():
         wells_in_meta.extend(wells_list)

    wells_in_meta = set(wells_in_meta)
    channels_in_meta = set([params_dict["channels"]])

    # Find missing wells and channels
    missing_wells = wells_in_meta - set_wells_found
    extra_wells = set_wells_found - wells_in_meta

    missing_channels = channels_in_meta - set_channels_found
    extra_channels = set_channels_found - channels_in_meta

    # Print missing and extra wells
    if missing_wells:
        print(f"The following wells {missing_wells} were listed in the metafile but not found in the image folder.")
    if extra_wells:
        print(f"The following wells {extra_wells} were found in the image folder but not listed on the metafile.")

    # Print missing and extra channels
    if missing_channels:
        print(f"The following channels {missing_channels} were listed in the metafile but not found in the image folder.")
    if extra_channels:
        print(f"The following channels {extra_channels} were found in the image folder but not listed on the metafile.")