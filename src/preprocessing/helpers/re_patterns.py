'''
This file contains re patterns that match filenames for different systems.
'''


import re

def get_file_pattern(system_name, file_type="frames"):


    pattern_dict = {
        "evos": {"raw": re.compile(
                        r'^(?P<device_id>[^_]+)_'
                        r'(?P<plate>Plate)_'
                        r'(?P<img_type>[A-Z]+)_.*_'
                        r'(?P<well_id>[A-Z]\d{2})f(?P<frame_id>\d{2})d(?P<channel_id>\d+)',
                        re.IGNORECASE),

                "roi":re.compile(
                r"^(?P<device_id>C\d+)_"
                     r"(?P<well_id>D\d+)_"
                     r"f(?P<frame_id>\d+)_"
                     r"r(?P<row_idx>\d+)_"
                     r"c(?P<col_idx>\d+)\.tif$",
                    re.IGNORECASE),

                "orig": re.compile(r'^(?P<device_id>[^_]+)_(?P<well_id>[^_]+)_f(?P<frame_id>[^_]+)\.tif$')
                },




        "cytation": {

            "raw": re.compile(
        r'^(?P<well_id>[A-Z]\d+)_'
            r'(?P<img_type_id>-?\d{1,2})_'
            r'(?P<channel_id>\d)_'
            r'(?P<frame_id>\d{1,4})_'
            r'(?P<channel>.+?)_'
            r'(?P<unknown2>\d{3})\.tif$',
            re.IGNORECASE),

            "roi":re.compile(
     r"^(?P<device_id>C\d+)_"
             r"(?P<well_id>D\d+)_"
             r"f(?P<frame_id>\d+)_"
             r"r(?P<row_idx>\d+)_"
             r"c(?P<col_idx>\d+)\.tif$",
            re.IGNORECASE),

            "orig": re.compile(r'^(?P<device_id>[^_]+)_(?P<well_id>[^_]+)_f(?P<frame_id>[^_]+)\.tif$'),

            "aligned": ''
        }

    }

    for key in pattern_dict.keys():
        if system_name.lower() == key:
            system_pattern = pattern_dict[key][file_type]
            return system_pattern