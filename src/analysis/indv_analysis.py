import os
import pdb
import sys

import numpy as np
import pandas as pd

from pathlib import Path
from typing import Union
from natsort import natsorted
from sympy.strategies.branch import condition

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from analysis.result_generation_helpers import generate_y_y_plot


def combine_and_summarize_well_preds(folder_path, out_dir, params_dict, well_id, file_suffix='_labelfree_counts.csv'):

    device_id = params_dict["device_ID"]
    cluster_thresh = params_dict.get('clusters_thresh', 10)
    colonies_thresh = params_dict.get('colony_thresh', 40)

    # create separate directories to save summary vs combined counts
    out_summary = out_dir / 'summary'
    out_comb = out_dir / 'raw_combined'
    os.makedirs(out_summary, exist_ok=True)
    os.makedirs(out_comb, exist_ok=True)

    filtered_files = list(folder_path.rglob(f'*{file_suffix}'))
    curr_sname = folder_path.name
    if not filtered_files:
        print("No matching CSV files found.")
        return

    df_list = []
    for file in filtered_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    if not df_list:
        print("No valid CSV files to combine.")
        return

    # get exp start date folder name
    start_sname = params_dict.get('start_sname', 'day1')
    combined_df = pd.concat(df_list, ignore_index=True)

    categorization_dict = {
        -1 : "noise/debris",
        0 : "empty",
        1 : "single",
        2 : "dublets",
        3 : "triplets",
        4 : "4"
    }

    bin_edges = [5, 10, 20, 30, 40]

    if curr_sname == start_sname:
        # Area-based results
        model_based_df = combined_df[['Well Center X', 'Well Center Y', 'Model-Based Count', 'Model-Based Category']].copy()
        model_based_df.dropna(axis=0, inplace=True)
        model_based_df.reset_index(drop=True, inplace=True)
        model_based_df.columns = ['Well Center X', 'Well Center Y', 'Total Cell Count', 'Well Category']

        combined_csv_filename = out_comb / f"{device_id}_{well_id}_{curr_sname}_raw_counts.csv"
        model_based_df.to_csv(combined_csv_filename, index=False)

        summary_fname = f"{device_id}_{well_id}_{curr_sname}_seeding_dist.xlsx"
        summary_csv_fpath = out_summary / summary_fname
        df = model_based_df[['Well Category', 'Total Cell Count']].copy()

        with pd.ExcelWriter(summary_csv_fpath, engine='openpyxl') as writer:
            # Well category distribution with percentages
            cat_counts = df['Well Category'].value_counts().reset_index()
            cat_counts.columns = ['Well Category', 'Count']
            total = cat_counts['Count'].sum()
            cat_counts['percent(%)'] = (cat_counts['Count'] / total * 100).round(2) if total > 0 else 0
            cat_counts.columns = ['Well Category', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
            cat_counts.to_excel(writer, sheet_name='categories', index=False)

            # Cell count distribution
            cell_counts = df['Total Cell Count'].value_counts().reset_index()
            cell_counts.columns = ['Total Cell Count', 'Count']
            total = cell_counts['Count'].sum()
            cell_counts['percent(%)'] = (cell_counts['Count'] / total * 100).round(2) if total > 0 else 0
            cell_counts.columns = ['Total Cell Count', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
            cell_counts.sort_values(by='Total Cell Count', inplace=True)


            title_suffix = "seeding distribution"
            summary_df =  generate_y_y_plot(cell_counts,
                              out_summary,
                              device_id, well_id, curr_sname,
                              title_suffix, categorization_dict,
                              summary_fname, bin_edges)

            summary_df.to_excel(writer, sheet_name='counts', index=False)

        print(f"Saved combined CSV, distributions, and categorized Excel to: {out_dir}")


    else:
        # Intensity-based results
        intensity_based_df = combined_df[['Well Center X', 'Well Center Y', 'Intensity-Based Count', 'Intensity-Based Category']].copy()
        intensity_based_df.dropna(axis=0, inplace=True)
        intensity_based_df.reset_index(drop=True, inplace=True)
        intensity_based_df.columns = ['Well Center X', 'Well Center Y', 'Total Cell Count', 'Well Category']

        combined_csv_filename = out_comb / f"{device_id}_{well_id}_{curr_sname}_raw_counts.csv"
        intensity_based_df.to_csv(combined_csv_filename, index=False)

        summary_fname = f"{device_id}_{well_id}_{curr_sname}_total_dist.xlsx"
        summary_csv_fpath = out_summary / summary_fname
        df = intensity_based_df[['Well Category', 'Total Cell Count']].copy()

        with pd.ExcelWriter(summary_csv_fpath, engine='openpyxl') as writer:
            # Well category distribution with percentages
            cat_counts = df['Well Category'].value_counts().reset_index()
            cat_counts.columns = ['Well Category', 'Count']
            total = cat_counts['Count'].sum()
            cat_counts['percent(%)'] = (cat_counts['Count'] / total * 100).round(2) if total > 0 else 0
            cat_counts.columns = ['Well Category', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
            cat_counts.to_excel(writer, sheet_name='categories', index=False)

            # Cell count distribution
            cell_counts = df['Total Cell Count'].value_counts().reset_index()
            cell_counts.columns = ['Total Cell Count', 'Count']
            total = cell_counts['Count'].sum()
            cell_counts['percent(%)'] = (cell_counts['Count'] / total * 100).round(2) if total > 0 else 0
            cell_counts.columns = ['Total Cell Count', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
            cell_counts.sort_values(by='Total Cell Count', inplace=True)

            title_suffix = "total well distribution"
            summary_df = generate_y_y_plot(cell_counts,
                              out_summary,
                              device_id, well_id, curr_sname,
                              title_suffix, categorization_dict,
                              summary_fname, bin_edges)

            summary_df.to_excel(writer, sheet_name='counts', index=False)


        # Create distributions for all occupied wells on termination day regardless of seeding day state
        # create array to store total clonogeninindex
        c_idx_arr = np.full((1, 1, len(bin_edges)), np.nan, dtype=float)
        summary_fname_nonempty = f"{device_id}_{well_id}_{curr_sname}_occ_dist.xlsx"
        summary_csv_fpath_nonempty = out_summary / summary_fname_nonempty

        with pd.ExcelWriter(summary_csv_fpath_nonempty, engine='openpyxl') as writer:

            # Cell count distribution
            nonempty_condition = df['Total Cell Count'] != 0
            cell_counts_nonempty = df['Total Cell Count'][nonempty_condition].value_counts().reset_index()
            cell_counts_nonempty.columns = ['Total Cell Count', 'Count']
            total_nonempty = cell_counts_nonempty['Count'].sum()
            cell_counts_nonempty['percent(%)'] = (cell_counts_nonempty['Count'] / total_nonempty * 100).round(2) if total_nonempty > 0 else 0
            cell_counts_nonempty.columns = ['Total Cell Count', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
            cell_counts_nonempty.sort_values(by='Total Cell Count', inplace=True)

            title_suffix = "total well distribution(occupied, no-tracking)"
            summary_df_nonempty = generate_y_y_plot(cell_counts_nonempty,
                                           out_summary,
                                           device_id, well_id, curr_sname,
                                           title_suffix, categorization_dict,
                                           summary_fname_nonempty, bin_edges)

            summary_df_nonempty.to_excel(writer, sheet_name='counts', index=False)

            if summary_df_nonempty is not None:

                summary_df_nonempty['lower_bound'] = pd.to_numeric(
                    summary_df_nonempty['Total Cell Count'].str.split('-', expand=True)[0],
                    errors='coerce'
                )

                # calculate clonogenic index based on different bin categories:
                for i, edge in enumerate(bin_edges):
                    mask = summary_df_nonempty['lower_bound'] >= edge
                    c_index = summary_df_nonempty.loc[mask, f'{curr_sname} percent(%)'].sum()
                    c_idx_arr[0, 0, i] = c_index

                # save clonogenic index arr
                arr_outpath = out_summary / f"{device_id}_{well_id}_cidx_allwells.npy"
                np.save(arr_outpath, c_idx_arr)

        print(f"Saved combined CSV, distributions, and categorized Excel to: {out_dir}")


def save_per_day_summary(well_fldpath: Union[Path, str],
         params_dict
)->None:
    well_id = well_fldpath.name
    counts_dir = well_fldpath / 'raw_counts'
    day_dirs = natsorted([d for d in counts_dir.iterdir() if d.is_dir()])
    if len(day_dirs) == 0:
        print(f"No day folders found in {well_fldpath}.")
        return

    # count cells for each day folder in the well folder
    for day_fldpath in day_dirs:
        if day_fldpath.is_dir():
            combine_and_summarize_well_preds(day_fldpath, well_fldpath, params_dict, well_id)

