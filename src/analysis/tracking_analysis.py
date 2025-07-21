import os
import pdb
import re
import sys
import glob
import numpy as np
import pandas as pd

from natsort import natsorted
from scipy.cluster.hierarchy import single
from scipy.spatial import cKDTree
from sympy.strategies.branch import condition

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from analysis.result_generation_helpers import generate_y_y_plot, clean_well_tracking_df, extract_id_and_cats, extract_id_and_counts

'''
Tracking workflow:
1. Counts are saved for each frame independent of the session. 
2. Aligned frames have the same raw counts filename across sessions.
3. For each frame, collect counts across two or more sessions and perform tracking by comparing well locations.
'''
def perform_well_tracking(well_path,
                          counts_dir,
                          device_id,
                          well_id,
                          file_suffix,
                          distance_thresh):

    # Get sorted list of day subdirectories (e.g., day1, day2, …)
    day_dirs = natsorted([d for d in counts_dir.iterdir() if d.is_dir()])
    if len(day_dirs) == 0:
        print(f"No day folders found in {well_path}.")
        return

    # Use the first day as reference for well positions
    reference_day_dir = day_dirs[0]
    # Find all files matching the suffix in the reference day
    ref_files = sorted(glob.glob(str(reference_day_dir / f'*{file_suffix}')))
    # Just keep the filenames (no path) to loop over
    ref_filenames = [os.path.basename(fpaths) for fpaths in ref_files]

    df_tracking = pd.DataFrame()  # will accumulate results across all files
    day_strdict = {}  # map day index to directory name for labeling columns

    # Loop over each reference CSV filename corresponding to each frame
    for framename in ref_filenames:
        all_centers = []     # list of center arrays per day
        all_counts = []      # list of count arrays per day
        all_categories = []  # list of category arrays per day
        ref_centers = None   # will hold centers from day 0

        # Read data for each day
        for day_idx, raw_dir in enumerate(day_dirs):
            day_strdict[day_idx] = raw_dir.name
            csv_path = raw_dir / framename
            if not os.path.exists(csv_path):
                # If file missing, pad lists with None
                all_centers.append(None)
                all_counts.append(None)
                all_categories.append(None)
                continue

            df = pd.read_csv(csv_path)
            centers = df[['Well Center X', 'Well Center Y']].values

            if day_idx == 0:
                # Day 0: use model-based counts/categories
                counts = df['Model-Based Count'].values
                categories = (df['Model-Based Category'].values
                              if 'Model-Based Category' in df.columns
                              else [None] * len(counts))
                ref_centers = centers
            else:
                # Other days: use intensity-based counts/categories
                counts = df['Intensity-Based Count'].values
                categories = (df['Intensity-Based Category'].values
                              if 'Intensity-Based Category' in df.columns
                              else [None] * len(counts))

            all_centers.append(centers)
            all_counts.append(counts)
            all_categories.append(categories)

        if ref_centers is None:
            # Skip this frame if reference day had no valid data
            continue

        num_wells = len(ref_centers)
        # Prepare arrays to hold matched results
        matched_counts = np.full((num_wells, len(day_dirs)), np.nan)
        matched_centers = [[None] * len(day_dirs) for _ in range(num_wells)]
        matched_categories = [[None] * len(day_dirs) for _ in range(num_wells)]

        # For each day, match wells to reference positions
        for day_idx, (centers, counts, cats) in enumerate(zip(all_centers, all_counts, all_categories)):
            if centers is None or counts is None:
                continue
            # Build a KD-tree for fast nearest-neighbor lookup
            tree = cKDTree(centers)
            # Query nearest neighbor within the distance threshold
            _, indices = tree.query(ref_centers, distance_upper_bound=distance_thresh)
            for i, idx in enumerate(indices):
                if idx < len(counts):
                    # Save matched count, center, and category
                    matched_counts[i][day_idx] = int(counts[idx].item())
                    x, y = centers[idx]
                    matched_centers[i][day_idx] = (x.item(), y.item())
                    matched_categories[i][day_idx] = cats[idx] if cats is not None else None

        # Build a combined DataFrame for this file
        combined_rows = []
        for i in range(num_wells):
            row = {
                'unique_well_id': f"{device_id}{well_id}W{i}",
                'source_file': framename,
            }
            # Add columns for each day’s location, count, and category
            for day_idx, day in enumerate(day_dirs):
                day_name = day_strdict[day_idx]
                row[f'{day_name}_well_loc'] = matched_centers[i][day_idx]
                row[f'{day_name}_counts'] = matched_counts[i][day_idx]
                row[f'{day_name}_category'] = matched_categories[i][day_idx]
            combined_rows.append(row)

        combined_df = pd.DataFrame(combined_rows)

        # Append to the overall tracking DataFrame
        df_tracking = pd.concat([df_tracking, combined_df], ignore_index=True)

    return df_tracking

def singles_analysis(counts_only_df, cats_only_df,
                      device_id,
                      well_id,
                      summary_texts,
                      out_summary):

    # get df where the cat for day 1 is singles
    cat_cols = natsorted([col for col in cats_only_df.columns if re.match(r"day\d+_category", col)])
    start_date_col_cat = cat_cols.pop(0)

    singles_condition = cats_only_df[start_date_col_cat] == 'single'
    singles_fate_df = cats_only_df[singles_condition].reset_index(drop=True)

    # change occurences of empty to 'died'
    for col in cat_cols:
        singles_fate_df.replace({col: {'empty': 'dead'}}, inplace=True)

    # store fate of singles for each day
    singles_dist_dfs = dict()

    # keep log for undetected singles
    for col in cat_cols:
        curr_sname = col.split('_')[0]

        # filter out data for detected singles
        singles_fate_dist = singles_fate_df[col].value_counts().reset_index().copy()
        total = singles_fate_dist['count'].sum()
        singles_fate_dist['percent(%)'] = (singles_fate_dist['count'] / total * 100).round(2) if total > 0 else 0
        singles_fate_dist.columns = ['Singles Fate', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
        singles_dist_dfs[col] = singles_fate_dist

    # REPEAT FOR COUNTS
    count_cols = natsorted([col for col in counts_only_df.columns if re.match(r"day\d+_count", col)])
    count_cols.pop(0)
    single_condition = cats_only_df[start_date_col_cat] == 'single'
    singles_fate_df_counts = counts_only_df[single_condition].reset_index(drop=True)

    # get total singles on start date
    num_singles_on_start = len(singles_fate_df_counts)

    categorization_dict = {
        -1: "undetected",
        0: "dead",
        1: "single",
        2: "dublets",
        3: "triplets",
        4: "4"
    }

    bin_edges = [5, 10, 20, 30, 40]

    # create array to store clonogeninindex
    c_idx_arr = np.full((1, 1, len(bin_edges)), np.nan, dtype=float)

    # Specify the output Excel file path
    summary_fname = f"{device_id}_{well_id}_singles_dist.xlsx"
    output_excel_path = out_summary / summary_fname

    # keep log for undetected singles
    for col in count_cols:
        curr_sname = col.split('_')[0]

        # get number of undetected singles
        undetected_condition = singles_fate_df_counts[col] == -1
        num_undetected = len(singles_fate_df_counts[col][undetected_condition])
        perc_undetected = round((num_undetected / num_singles_on_start), 4) * 100 if num_undetected > 0 else 0
        txt = f"Undetected singles {col.split('_')[0]}: {num_undetected} ({perc_undetected:.2f})%\n"
        summary_texts.append(txt)

        # filter out data for detected singles
        singles_fate_dist_counts = singles_fate_df_counts[col].value_counts().reset_index().copy()
        total = singles_fate_dist_counts['count'].sum()
        singles_fate_dist_counts['percent(%)'] = (singles_fate_dist_counts['count'] / total * 100).round(3) if total > 0 else 0
        singles_fate_dist_counts.columns = ['Singles Fate', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
        singles_fate_dist_counts.sort_values(by='Singles Fate', inplace=True)

        title_suffix = "singles fate distribution"
        summary_df = generate_y_y_plot(singles_fate_dist_counts,
                                       out_summary,
                                       device_id, well_id, curr_sname,
                                       title_suffix, categorization_dict,
                                       summary_fname,
                                       bin_edges,
                                       bar_color='#86AAE7')

        if summary_df is not None:
            singles_dist_dfs[col] = summary_df.copy()

            summary_df['lower_bound'] = pd.to_numeric(
                summary_df['Singles Fate'].str.split('-', expand=True)[0],
                errors='coerce'
            )

            summary_texts.append("\nClonogenic Index Info:\n")
            # calculate clonogenic index based on different bin categories:
            for i, edge in enumerate(bin_edges):
                mask = summary_df['lower_bound'] >= edge
                c_index = summary_df.loc[mask, f'{curr_sname} percent(%)'].sum()
                c_idx_arr[0, 0, i] = c_index
                txt = f"Clonogeninc Index({edge}+ cells): {c_index:.4f}%\n"
                summary_texts.append(txt)

    # save clonogenic index arr
    arr_outpath = out_summary / f"{device_id}_{well_id}_cidx.npy"
    np.save(arr_outpath, c_idx_arr)

    # Write all single-cell distribution DataFrames to one Excel file with multiple sheets
    if len(singles_dist_dfs) > 0:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            for key, val in singles_dist_dfs.items():
                sheet_name = key[:31]  # Excel limits sheet names to 31 characters
                val.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved single-cell distributions to: {output_excel_path}")

        summary_txt_path = out_summary / f"{device_id}_{well_id}_tracking_logs.txt"

        with open(summary_txt_path, 'w') as f:
            for line in summary_texts:
                f.write(line)
    else:
        print(f"Singles df could not be saved as the distribution is empty: {output_excel_path}")


def non_empty_analysis(cleaned_df,
                    device_id,
                    well_id,
                    out_summary):

    counts_only_df = extract_id_and_counts(cleaned_df)

    # get column names containing counts
    count_cols = natsorted([col for col in counts_only_df.columns if re.match(r"day\d+_count", col)])
    start_date_col = count_cols.pop(0)  # remove start date count col

    categorization_dict = {
        -1: "undetected",
        0: "dead",
        1: "single",
        2: "dublets",
        3: "triplets",
        4: "4"
    }

    bin_edges = [5, 10, 20, 30, 40]

    # Specify the output Excel file path
    occ_w_ghosts_dist_dfs = dict()    # termination day dist with tracking, has info on dead cells
    summary_fname_track = f"{device_id}_{well_id}_ghosts_occ_dist.xlsx"
    output_excel_path_track = out_summary / summary_fname_track

    for col in count_cols:

        curr_sname = col.split('_')[0]

        # filter out empty to empty
        empt2empt_condition = (
                (counts_only_df[start_date_col] == 0) &
                (counts_only_df[col] == 0)
            )

        # pdb.set_trace()
        # get value counts for wells that did not start out empty
        e2e_counts_df = counts_only_df[col][~empt2empt_condition].value_counts().reset_index().copy()
        total = e2e_counts_df['count'].sum()
        e2e_counts_df['percent(%)'] = (e2e_counts_df['count'] / total * 100).round(2) if total > 0 else 0
        e2e_counts_df.columns = ['Total Cell Count', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
        e2e_counts_df.sort_values(by='Total Cell Count', inplace=True)

        title_suffix = "overall dist of all occupied wells"
        summary_df = generate_y_y_plot(e2e_counts_df,
                                       out_summary,
                                       device_id, well_id, curr_sname,
                                       title_suffix, categorization_dict,
                                       f"{device_id}_{well_id}_{curr_sname}_ghosts_occ_dist.xlsx",
                                       bin_edges)
        if summary_df is not None:
            occ_w_ghosts_dist_dfs[col] = summary_df


    if len(occ_w_ghosts_dist_dfs) > 0:
        with pd.ExcelWriter(output_excel_path_track, engine='openpyxl') as writer:
            for key, val in occ_w_ghosts_dist_dfs.items():
                sheet_name = key[:31]  # Excel limits sheet names to 31 characters
                val.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved all occupied wells with tracking distributions to: {output_excel_path_track}")
    else:
        print(f"Occupied wells df could not be saved as the distribution is empty: {output_excel_path_track}")

    # OVERALL TERMINATION DAY DIST DAY-1 NON-EMPYU
    strt_occ_wells_df = dict()  # termination day all occupied wells, does not have info for cells that died
    summary_fname_occ = f"{device_id}_{well_id}_strt_occ_dist.xlsx"
    output_excel_path_occ = out_summary / summary_fname_occ

    for col in count_cols:
        curr_sname = col.split('_')[0]

        # filter out wells that were occupied on start date
        strt_nonempty_condition = counts_only_df[start_date_col] == 0

        # get value counts for wells that did not start out empty
        occ_counts_df = counts_only_df[col][~ strt_nonempty_condition].value_counts().reset_index().copy()
        total = occ_counts_df['count'].sum()
        occ_counts_df['percent(%)'] = (occ_counts_df['count'] / total * 100).round(2) if total > 0 else 0
        occ_counts_df.columns = ['Total Cell Count', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
        occ_counts_df.sort_values(by='Total Cell Count', inplace=True)

        title_suffix = "overall dist of all occupied wells on day0/1"
        summary_df = generate_y_y_plot(occ_counts_df,
                                       out_summary,
                                       device_id, well_id, curr_sname,
                                       title_suffix, categorization_dict,
                                       f"{device_id}_{well_id}_{curr_sname}_strt_occ_dist.xlsx",
                                       bin_edges)

        if summary_df is not None:
            strt_occ_wells_df[col] = summary_df

    if len(strt_occ_wells_df) > 0:
        with pd.ExcelWriter(output_excel_path_occ, engine='openpyxl') as writer:
            for key, val in strt_occ_wells_df.items():
                sheet_name = key[:31]  # Excel limits sheet names to 31 characters
                val.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved occupied wells distributions withour tracking to: {output_excel_path_occ}")
    else:
        print(f"Ocuupied wells df could not be saved as the distribution is empty: {output_excel_path_occ}")


def clones_analysis(counts_only_df,
                    device_id,
                    well_id,
                    out_summary,
                    min_clones_thresh=5):

    # get column names that contain counts
    count_cols = natsorted([col for col in counts_only_df.columns if re.match(r"day\d+_count", col)])
    start_date_col = count_cols.pop(0)   # remote start day col

    categorization_dict = {
        -1: "noise/debris",
        0: "empty",
        1: "single",
        2: "dublets",
        3: "triplets",
        4: "4"
    }

    bin_edges = [5, 10, 20, 30, 40]

    # Specify the output Excel file path
    summary_fname = f"{device_id}_{well_id}_clones_dist.xlsx"
    output_excel_path = out_summary / summary_fname

    clones_dist_dfs = dict()

    # keep log for undetected singles
    for col in count_cols:
        curr_sname = col.split('_')[0]

        # wells with more than 'min_clones_thresh' for that session
        clones_condition = counts_only_df[col] >= min_clones_thresh

        # get start date counts df
        start_date_counts_df = counts_only_df[start_date_col][clones_condition]

        start_date_counts_dist = start_date_counts_df.value_counts().reset_index().copy()
        total = start_date_counts_dist['count'].sum()
        start_date_counts_dist['percent(%)'] = (start_date_counts_dist['count'] / total * 100).round(2) if total > 0 else 0
        start_date_counts_dist.columns = ['Num Cells at Start', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
        start_date_counts_dist.sort_values(by='Num Cells at Start', inplace=True)

        title_suffix = "total clones distribution"
        # print(start_date_counts_dist)
        summary_df = generate_y_y_plot(start_date_counts_dist,
                                       out_summary,
                                       device_id, well_id, curr_sname,
                                       title_suffix, categorization_dict,
                                       f"{device_id}_{well_id}_{curr_sname}_ghost_clones_dist.xlsx",
                                       bin_edges,
                                       bar_color='#F0AB9E')

        if summary_df is not None:
            clones_dist_dfs[col] = summary_df.copy()

    # Write all single-cell distribution DataFrames to one Excel file with multiple sheets
    if len( clones_dist_dfs) > 0:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            for key, val in  clones_dist_dfs.items():
                sheet_name = key[:31]  # Excel limits sheet names to 31 characters
                val.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"Saved clones distributions to: {output_excel_path}")
    else:
        print(f"Clones df could not be saved as the distribution is empty: {output_excel_path}")

    # repeat for clones that came out of wells occupied on start date
    for col in count_cols:
        curr_sname = col.split('_')[0]

        # wells with more than 'min_clones_thresh' for that session
        clones_condition = (counts_only_df[col] >= min_clones_thresh) & (counts_only_df[start_date_col] != 0)

        # get start date counts df
        start_date_counts_df = counts_only_df[start_date_col][clones_condition]

        start_date_counts_dist = start_date_counts_df.value_counts().reset_index().copy()
        total = start_date_counts_dist['count'].sum()
        start_date_counts_dist['percent(%)'] = (start_date_counts_dist['count'] / total * 100).round(2) if total > 0 else 0
        start_date_counts_dist.columns = ['Num Cells at Start', f'{curr_sname} Num Wells', f'{curr_sname} percent(%)']
        start_date_counts_dist.sort_values(by='Num Cells at Start', inplace=True)

        title_suffix = "total clones distribution for wells occupied on day0/1"
        # print(start_date_counts_dist)
        summary_df = generate_y_y_plot(start_date_counts_dist,
                                       out_summary,
                                       device_id, well_id, curr_sname,
                                       title_suffix, categorization_dict,
                                       f"{device_id}_{well_id}_{curr_sname}_clones_dist.xlsx",
                                       bin_edges,
                                       bar_color='#F0AB9E')

        if summary_df is not None:
            clones_dist_dfs[col] = summary_df.copy()

    # Write all single-cell distribution DataFrames to one Excel file with multiple sheets
    if len(clones_dist_dfs) > 0:
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            for key, val in clones_dist_dfs.items():
                sheet_name = key[:31]  # Excel limits sheet names to 31 characters
                val.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Saved clones distributions to: {output_excel_path}")
    else:
        print(f"Clones df could not be saved as the distribution is empty: {output_excel_path}")

def process_well_predictions(well_path, file_suffix='_dual_counts.csv', distance_thresh=30):

    # cluster_thresh = params_dict.get('clusters_thresh', 10)
    # colonies_thresh = params_dict.get('colony_thresh', 40)
    well_id = well_path.name
    device_id = well_path.parents[2].name
    counts_dir = well_path / 'raw_counts'

    out_summary = well_path / 'summary'
    out_comb = well_path / 'raw_combined'
    os.makedirs(out_summary, exist_ok=True)
    os.makedirs(out_comb, exist_ok=True)

    # df with tracking
    combined_df_all = perform_well_tracking(well_path, counts_dir, device_id, well_id,
                                        file_suffix, distance_thresh)


    # add (-1, -1) for missing locations and category 'undetected' for missing well categories
    cleaned_df, summary_texts = clean_well_tracking_df(combined_df_all, device_id, well_id)

    # Save combined file
    if not cleaned_df.empty:
        combined_save_path = os.path.join(out_comb, f"{device_id}_{well_id}_tracking_raw.xlsx")
        cleaned_df.to_excel(combined_save_path, index=False)
        print(f"Saved combined file: {combined_save_path}")


    # extract counts only dataframe and save it for further analysis
    counts_only_df = extract_id_and_counts(cleaned_df)
    if not counts_only_df.empty:
        combined_save_path = os.path.join(out_comb, f"{device_id}_{well_id}_tracking_raw_binned_counts_only.xlsx")
        counts_only_df.to_excel(combined_save_path, index=False)
        print(f"Saved combined file: {combined_save_path}")

    # get category only df
    cats_only_df = extract_id_and_cats(cleaned_df)
    if not cats_only_df.empty:
        combined_save_path = os.path.join(out_comb, f"{device_id}_{well_id}_tracking_raw_categories_only.xlsx")
        cats_only_df.to_excel(combined_save_path, index=False)
        print(f"Saved combined file: {combined_save_path}")

    # save overall distribution for occupied wells
    non_empty_analysis(cleaned_df, device_id, well_id, out_summary)
    singles_analysis(counts_only_df, cats_only_df, device_id, well_id, summary_texts, out_summary)
    clones_analysis(counts_only_df, device_id, well_id, out_summary)


