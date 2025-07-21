import sys

def flip_dict(params_dict):
    flipped_dict = {}

    try:
        # Check if "device_to_well_dict" exists in params_dict
        device_to_well_dict = params_dict["device_to_well_dict"]

        # Iterate over each key-value pair in the input dictionary
        for device, wells in device_to_well_dict.items():
            for well in wells:
                flipped_dict[well] = device

    except KeyError:
        print("Error: 'device_to_well_dict' key is missing in params_dict. Returning None value.")

        return None
    return flipped_dict


def create_well_translation_dict(params_dict):
    try:
        # Ensure required keys exist
        device_to_wells_dict = params_dict["device_to_well_dict"]
        well_id_translation = params_dict["well_ID_translation"]

        # Flatten the wells from all devices into a single list
        n_devices = len(device_to_wells_dict)
        all_wells = [well for wells in device_to_wells_dict.values() for well in wells]
        well_id_translation = well_id_translation * n_devices
        well_translation_dict = {well: translation for well, translation in zip(all_wells, well_id_translation)}

        return well_translation_dict

    except KeyError as e:
        print(f"Missing key(s) 'device_to_well_dict', 'well_ID_translation' in params_dict: {e}. Returning None value.")
        return None

