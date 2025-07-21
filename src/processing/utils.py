import pdb

import pandas as pd
from pathlib import Path
import re

def combine_and_summarize_well_preds(folder_path, params_dict):
    exp_date = params_dict["exp_date"]
    device_id = params_dict["device_ID"]
    well_id = folder_path.parents[1].name
    folder_path = Path(folder_path) / 'count_results'
    dest_folder_path = Path(folder_path) / 'count_results' / well_id
    matching_files = list(folder_path.rglob("*_well_preds.csv"))
    # pdb.set_trace()

    # Filter files like 'deviceID_wellID_xxx_well_preds.csv'
    filtered_files = [f for f in matching_files if re.match(r"^[^_]+_[^_]+_.*_well_preds\.csv$", f.name)]

    if not filtered_files:
        print("No matching CSV files found.")
        return

    # Extract device_id_well_id from a sample filename
    match = re.match(r"^([^_]+_[^_]+)_.*_well_preds\.csv$", filtered_files[0].stem)
    if not match:
        print("Could not parse device_id and well_id from filename.")
        return

    device_well_id = match.group(1)

    # Combine all matched CSVs
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

    combined_df = pd.concat(df_list, ignore_index=True)

    # --- Processing ---
    # Drop 'single' rows with negative cell count
    mask_single_negative = (combined_df['Well Catgory'] == 'single') & (combined_df['Total Cell Count'] < 0)
    combined_df = combined_df[~mask_single_negative]

    # Set non-single negative counts to -1
    mask_non_single_negative = (combined_df['Well Catgory'] == 'non-single') & (combined_df['Total Cell Count'] < 0)
    combined_df.loc[mask_non_single_negative, 'Total Cell Count'] = -1

    # Save cleaned combined CSV
    combined_csv_filename = folder_path / f"{device_well_id}_combined_well_preds.csv"
    combined_df.to_csv(combined_csv_filename, index=False)

    # Keep only necessary columns
    summary_df = combined_df[['Well Catgory', 'Total Cell Count']]

    # Save well category distribution
    cat_counts = summary_df['Well Catgory'].value_counts().reset_index()
    cat_counts.columns = ['Well Catgory', 'Count']
    cat_counts.to_csv(folder_path / f"{device_well_id}_well_category_dist.csv", index=False)

    # Save total cell count distribution
    cell_counts = summary_df['Total Cell Count'].value_counts().reset_index()
    cell_counts.columns = ['Total Cell Count', 'Count']
    cell_counts.to_csv(folder_path / f"{device_well_id}_cell_count_per_well_dist.csv", index=False)

    # --- Categorized Distribution ---
    def categorize_count(val):
        if val == 0:
            return 'empty'
        elif val == 1:
            return 'singles'
        elif val == 2:
            return 'doublets'
        elif val == 3:
            return 'triplets'
        elif val == 4:
            return '4'
        elif val == -1:
            return 'noise/debris'
        elif val >= 5:
            return '5+'
        else:
            return 'unknown'

    summary_df['Cell Count Category'] = summary_df['Total Cell Count'].apply(categorize_count)
    category_dist = summary_df['Cell Count Category'].value_counts().reset_index()
    category_dist.columns = ['Cell Count Category', 'Count']

    # Save to Excel
    excel_path = folder_path / f"{device_well_id}_well_category_count_dist.xlsx"
    category_dist.to_excel(excel_path, index=False)

    print(f"Saved combined CSV, distributions, and categorized Excel to: {folder_path}")



if __name__ == "__main__":

    folder_paths = [
        'data/organized/04_29_25_cytation_SCB_vib-cont-seeding_C346_multiday_U87_2025-04-30_14-41-48/C346/cropped_aligned/D6/day1'
    ]

    for fpath in folder_paths:
        combine_and_summarize_well_preds(fpath)