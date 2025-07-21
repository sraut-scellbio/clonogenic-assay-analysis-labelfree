import pdb
import os
import pickle
import re
import bisect
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from natsort import natsorted
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator

def extract_id_and_counts(df):
    # Identify all dayX_counts columns
    count_cols = [col for col in df.columns if re.match(r"day\d+_counts", col)]

    # Required base columns
    base_cols = ['unique_well_id', 'source_file']

    # Return filtered DataFrame
    return df[base_cols + count_cols].copy()

def extract_id_and_cats(df):
    # Identify all dayX_counts columns
    cat_cols = [col for col in df.columns if re.match(r"day\d+_category", col)]

    # Required base columns
    base_cols = ['unique_well_id', 'source_file']

    # Return filtered DataFrame
    return df[base_cols + cat_cols].copy()

def clean_well_tracking_df(df, device_id=None, well_id=None):

    summary_texts = []
    txt = f"Tracking Summary for {device_id} well {well_id}:\n"
    summary_texts.append(txt)

    original_len = len(df)

    # # Step 1: Remove rows where day1_category is 'non-single' but day1_counts is 1
    # condition = (df['day1_category'] == 'non-single') & (df['day1_counts'] == 1)
    # df = df[~condition].copy()

    # Step 2: Replace negative counts with 0 for all *_counts columns
    count_cols = natsorted([col for col in df.columns if re.match(r"day\d+_counts", col)])
    loc_cols = natsorted([col for col in df.columns if re.match(r"day\d+_well_loc", col)])
    cat_cols = natsorted([col for col in df.columns if re.match(r"day\d+_category", col)])

    for col in count_cols:
        df[col] = df[col].astype('Int64')

    count_cols.pop(0)
    loc_cols.pop(0)
    cat_cols.pop(0)

    # Step 3: Count and report NaNs for each day
    for col in count_cols:
        nan_count = df[col].isna().sum()
        percent = (nan_count / len(df)) * 100 if len(df) > 0 else 0
        day_label = col.split('_')[0]
        txt = f"Number of wells undetected on {day_label}: {nan_count} ({percent:.2f}%)\n"
        summary_texts.append(txt)

    # Step 4: Drop rows with any NaN in *_counts columns
    before_drop = len(df)
    after_drop = len(df.dropna(subset=count_cols).copy())
    dropped = before_drop - after_drop
    dropped_percent = (dropped / before_drop) * 100 if before_drop > 0 else 0
    txt = f"Number of wells that could not be tracked: {dropped} ({dropped_percent:.2f}%)\n"
    summary_texts.append(txt)

    # fill null values for undetected wells
    for col in cat_cols:
        df[col] = df[col].fillna('undetected')

    for col in count_cols:
        df[col] = df[col].fillna(-1)

    for col in loc_cols:
        df[col] = df[col].apply(lambda x: x if pd.notnull(x) else (-1, -1))

    # Step 5: Reset index
    df = df.reset_index(drop=True)
    return df, summary_texts


def add_well_shifts(df):
    if 'day1_well_loc' not in df.columns:
        raise ValueError("Column 'day1_well_loc' not found in the dataframe.")

    # Identify all dayX_well_loc columns (excluding day1)
    day_loc_cols = [col for col in df.columns if re.match(r"day\d+_well_loc", col) and not col.startswith("day1")]

    for col in day_loc_cols:
        day = col.split('_')[0]
        shift_col = f"{day}_shift"

        # Compute Euclidean distance between dayX_well_loc and day1_well_loc
        df[shift_col] = df.apply(
            lambda row: np.linalg.norm(np.array(row[col]) - np.array(row['day1_well_loc']))
            if pd.notnull(row[col]) and pd.notnull(row['day1_well_loc']) else np.nan,
            axis=1
        )

    return df

def generate_y_y_plot(df, out_dir,
                      device_id='N/A',
                      well_id='N/A',
                      sname='N/A',
                      title_suffix='',
                      categorization_dict=None,
                      filename=None,
                      bin_edges=None,
                      bar_color='#1b6cdc'):
    if len(df) == 0:
        print(f"Summary Distribution Dataframe for {filename} is empty. Returning None\n")
        return None

    # Unpack columns
    x, y1, y2 = tuple(df.columns)
    y1_label = 'Num Wells'
    x_label = x

    filename = filename.split('.')[0] + '.png'

    # Step 1: Extract set of reserved keys (e.g., -1 to 4)
    reserved = set(categorization_dict.keys()) if categorization_dict else set()

    # Step 2: Bin values
    def bin_or_keep(val):
        if val in reserved:
            return val  # keep reserved as-is
        else:
            # Find which bin this value belongs to
            idx = bisect.bisect_right(bin_edges, val)
            if idx == 0:
                return f"<{bin_edges[0]}"
            elif idx == len(bin_edges):
                return f"{bin_edges[-1]}+"
            else:
                lower = bin_edges[idx - 1]
                upper = bin_edges[idx] - 1
                return f"{lower}-{upper}"

    def sort_key(val):
        try:
            return int(val)  # for values like -1, 0, 1, etc.
        except ValueError:
            match = re.match(r"(\d+)-\d+", val)
            if match:
                return int(match.group(1))
            else:
                return float('inf')

    # Apply binning
    df[x] = df[x].apply(bin_or_keep)

    # Convert to string for display consistency
    df[x] = df[x].astype(str)

    # Group and re-aggregate counts and percentages
    df = df.groupby(x, as_index=False).agg({y1: 'sum', y2: 'sum'})
    df.sort_values(by=x, key=lambda col: col.map(sort_key), inplace=True)

    # Map reserved values using categorization_dict
    if categorization_dict:
        str_mapping = {str(k): v for k, v in categorization_dict.items()}
        df[x] = df[x].map(lambda val: str_mapping.get(str(val), val))

    # Title
    title = f"{device_id} Well {well_id} Session {sname}: {title_suffix}"

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df[x], df[y1], alpha=0.85, color=bar_color, edgecolor='gray', label=y1_label)
    ax.set_ylim(0, df[y1].max() * 1.10)

    # Add count and % labels stacked vertically
    for bar, pct in zip(bars, df[y2]):
        count = int(bar.get_height())
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = count + max(df[y1]) * 0.04
        ax.text(x_pos, y_pos, f'{count}', ha='center', va='bottom', fontsize=8)
        ax.text(x_pos, count + max(df[y1]) * 0.01, f'{pct:.2f}%', ha='center', va='bottom', fontsize=8, weight='bold')

    # Axes formatting
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y1_label, fontsize=11)
    ax.set_title(title, fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Y-axis spacing
    intr = (df[y1].max() / 8)
    rounded_intr = int(round(intr / 10) * 10)
    if rounded_intr > 0:
        ax.yaxis.set_major_locator(MultipleLocator(rounded_intr))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

    # Grid and layout
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # Save bar chart
    out_path = os.path.join(out_dir, filename or f"{sname}_cell_count_distribution.png")
    plt.savefig(out_path)
    plt.close()

    return df


def generate_sample_clonogenic_index_plate(
    max_pct=15,
    row_decay=1.2,
    col_decay=1.5,
    replicate_cols=((1,2,3), (4,5,6), (7,8,9), (10,11,12)),
    replicate_rows=((0,7),),   # 0-based A<->H
    noise_std=0.7              # standard deviation of added noise (in percent)
):
    """
    Create an 8×12×1 plate of mock clonogenic-index percentages:
      • Start at `max_pct` in A1, subtract `row_decay` per row and `col_decay` per column.
      • Columns in each tuple of `replicate_cols` share the same base values.
      • Rows in each tuple of `replicate_rows` share the same base values.
      • Finally adds Gaussian noise (mean=0, std=noise_std) to *every* well, then clips to [0, max_pct].

    Returns
    -------
    data3d : np.ndarray, shape (8,12,1)
        The simulated clonogenic index (%) for wells A1–H12, with a singleton “channel” dimension.
    row_labels : list[str]
        ['A','B','C','D','E','F','G','H']
    col_labels : list[int]
        [1,2,…,12]
    """
    rows, cols = 8, 12

    # 1) Build the base gradient
    data = np.zeros((rows, cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            data[i, j] = max_pct - i*row_decay - j*col_decay
    data = np.clip(data, 0, None)

    # 2) Enforce column replicates
    for grp in replicate_cols:
        idxs = [c-1 for c in grp]
        base = data[:, idxs[0]].copy()
        for c in idxs[1:]:
            data[:, c] = base

    # 3) Enforce row replicates (e.g. A <-> H)
    for r1, r2 in replicate_rows:
        data[r2, :] = data[r1, :]

    # 4) Add per‐well random noise, then clip to valid range
    noise = np.random.normal(loc=0.0, scale=noise_std, size=data.shape)
    data_noisy = np.clip(data + noise, 0.0, max_pct)

    # 5) Add singleton channel dimension
    data3d = data_noisy[..., np.newaxis]  # shape (8,12,1)

    row_labels = list("ABCDEFGH")
    col_labels = list(range(1, cols+1))
    return data3d, row_labels, col_labels


def create_array_heatmap(results_summary_fpath,
                                device_id,
                                pattern,
                                cmap=None,
                                max_norm=None,
                                suffix="cindex_heatmap",
                                show_cbar=True,
                                bin_edges=(5, 10, 20, 30, 40)):
    """
    Scans folder for files like "{device_id}_{well_id}_cidx.npy",
    combines them into a (rows, cols, len(bin_edges)) array of percentages,
    then for each bin-edge channel creates a single global–normalized heatmap
    (annotated with the raw pct values) and saves it as a PNG.
    """
    # 1) combine logic
    files = []
    for fname in os.listdir(results_summary_fpath):
        m = pattern.match(fname)
        if not m:
            continue
        row, col = m.group(1), int(m.group(2))
        files.append((fname, row, col))

    if not files:
        raise ValueError("No matching .npy files found.")

    row_labels = sorted({r for _, r, _ in files})
    col_labels = sorted({c for _, _, c in files})
    row_to_i  = {r:i for i,r in enumerate(row_labels)}
    col_to_j  = {c:j for j,c in enumerate(col_labels)}

    sample = np.load(os.path.join(results_summary_fpath, files[0][0]))
    if sample.shape[:2] != (1,1):
        raise ValueError(f"Expected (1,1,n)-shaped files, got {sample.shape}")
    n_channels = sample.shape[2]

    combined = np.full((len(row_labels), len(col_labels), n_channels), np.nan,
                        dtype=sample.dtype)
    for fname, row, col in files:
        arr = np.load(os.path.join(results_summary_fpath, fname))
        combined[row_to_i[row], col_to_j[col], :] = arr.reshape(n_channels)


    # combined, _, _ = generate_sample_clonogenic_index_plate()

    # ensure output folder exists
    os.makedirs(results_summary_fpath, exist_ok=True)

    for idx, edge in enumerate(bin_edges):
        raw = combined[:, :, idx]

        if max_norm is None:
            # per‐channel min/max
            mn = np.nanmin(raw)
            mx = np.nanmax(raw)

        else:
            # fixed min-max
            mn = 0
            mx = max_norm

        # if min and max value are the same, don't generate a heatmap
        if mn == mx == 0:
            continue
        denom = (mx - mn) or 1.0
        norm = (raw - mn) / denom

        mask = np.isnan(raw)

        plt.figure(figsize=(len(col_labels) * 1 + 1,
                            len(row_labels) * 1 + 1))
        ax = sns.heatmap(
            norm,
            mask=mask,
            annot=raw,
            fmt=".2f",
            cmap=cmap,
            vmin=0,
            vmax=1,
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot_kws={"fontsize": 12},
            cbar=show_cbar,
            square=True
        )
        # increase sine fo ticklabels
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)

        if show_cbar:
            # custom cbar labeling
            cbar = ax.collections[0].colorbar
            # ticks = np.linspace(0, 1, 5)
            # real_ticks = np.round(mn + ticks * (mx - mn), 1)
            # cbar.set_ticks(ticks)
            # cbar.set_ticklabels(real_ticks)
            # cbar.ax.tick_params(labelsize=9)
            cbar.set_label(f"Percent (%)", fontsize=9)

        ax.set_title(f"Clonogenic index\n (colonies = {edge}+ cells)", fontsize=9)
        ax.set_xlabel("Well Column", fontsize=9)
        ax.set_ylabel("Well Row", fontsize=9)
        plt.tight_layout()

        # save and close
        plt.show()
        fig_name = f"{device_id}_{suffix}_{edge}+.png"
        out_path = os.path.join(results_summary_fpath, fig_name)
        # plt.savefig(out_path, dpi=150)
        plt.close()

    print(f"Saved {len(bin_edges)} heatmaps to {results_summary_fpath}")


def create_intensity_heatmap(device_id,
                             results_dict,
                             params_dict,
                             output_dir=".",
                             cmap=None,
                             textsize=9,
                             show_cbar=True,
                             split=False,
                             cell_lines=None):
    """
    Generates heatmap(s) for a 96‑well plate:
      - if split=False: one heatmap (cols 1–12)
      - if split=True: two subplots (cols 1–6 | cols 7–12), labeled by cell_lines tuple

    Args:
        device_id (str)
        results_dict (dict): { well_id: np.array shape (1,1,n_sessions) }
        params_dict (dict): must contain "data_folders" dict for session names
        output_dir (str)
        cmap: matplotlib colormap
        textsize (int)
        show_cbar (bool)
        split (bool): whether to split into two half‑plate heatmaps
        cell_lines (tuple of str): (left_half_label, right_half_label)
    """

    def short_formatter(x, pos):
        real = mn + x * (mx - mn)
        if real >= 1e9:
            return f"{real / 1e9:.1f}B"
        elif real >= 1e6:
            return f"{real / 1e6:.1f}M"
        else:
            return f"{real:,.0f}"

    os.makedirs(output_dir, exist_ok=True)

    # 1) load/save dict
    pkl_path = os.path.join(output_dir,
                            f"{device_id}_total_intensity_results_dict.pkl")
    if results_dict is None:
        with open(pkl_path, "rb") as f:
            results_dict = pickle.load(f)
        print(f"Loaded results dict from {pkl_path}")
        if not results_dict:
            print("Intensity dict empty—no heatmaps generated.")
            return
    else:
        with open(pkl_path, "wb") as f:
            pickle.dump(results_dict, f)
        print(f"Saved results dict to {pkl_path}")

    # 2) session names
    try:
        session_names = list(params_dict["data_folders"].keys())
    except Exception:
        n_s = next(iter(results_dict.values())).shape[2]
        session_names = [f"S{i+1}" for i in range(n_s)]
        print("Warning: no data_folders in params; using", session_names)

    # 3) row/col labels
    row_labels = sorted({wid[0] for wid in results_dict})
    col_labels = sorted({int(wid[1:]) for wid in results_dict})
    row_to_i = {r:i for i,r in enumerate(row_labels)}
    col_to_j = {c:j for j,c in enumerate(col_labels)}

    # 4) assemble full array
    sample = next(iter(results_dict.values()))
    n_sessions = sample.shape[2]
    combined = np.full((len(row_labels), len(col_labels), n_sessions),
                       np.nan, dtype=float)
    for well_id, arr in results_dict.items():
        r, c = well_id[0], int(well_id[1:])
        combined[row_to_i[r], col_to_j[c], :] = arr.reshape(n_sessions)

    # save array as a .csv file
    out_xlsx = os.path.join(output_dir, f"{device_id}_intensity_heatmap_values16.xlsx")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for idx, sess in enumerate(session_names):
            # extract the (row × col) 2D for this session
            arr2d = combined[:, :, idx]
            # build a DataFrame with your row/col labels
            df = pd.DataFrame(
                arr2d,
                index=row_labels,
                columns=col_labels
            )
            # write to a sheet named exactly like your session
            df.to_excel(writer, sheet_name=sess)

    print(f"Wrote {len(session_names)} sheets to {out_xlsx}")

    # default cell line labels
    if split and (not cell_lines or len(cell_lines) != 2):
        cell_lines = ("Left", "Right")

    # 5) loop sessions
    for idx in range(n_sessions):
        raw = combined[:, :, idx]
        if np.all(np.isnan(raw)):
            sess = session_names[idx] if idx < len(session_names) else f"S{idx+1}"
            print(f"Session '{sess}' all-NaN; skipping.")
            continue

        session = session_names[idx] if idx < len(session_names) else f"S{idx+1}"

        if not split:
            # single full‐plate heatmap
            mn, mx = np.nanmin(raw), np.nanmax(raw)
            denom = (mx - mn) or 1.0
            norm = (raw - mn) / denom
            mask = np.isnan(raw)

            plt.figure(figsize=(len(col_labels)+1, len(row_labels)+1))
            ax = sns.heatmap(norm, mask=mask, annot=norm, fmt=".3f",
                             cmap=cmap, vmin=0, vmax=1,
                             xticklabels=col_labels, yticklabels=row_labels,
                             annot_kws={"fontsize":textsize},
                             cbar=show_cbar, square=True)
            plt.xticks(fontsize=textsize)
            plt.yticks(fontsize=textsize)

            if show_cbar:
                cbar = ax.collections[0].colorbar

                # 1) pick 5 evenly spaced ticks in normalized [0,1] space
                ticks = np.linspace(0, 1, 5)
                cbar.set_ticks(ticks)

                cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(short_formatter))
                cbar.ax.tick_params(labelsize=textsize)
                cbar.set_label("Intensity (×10⁶)", fontsize=int(textsize * 1.2))

            ax.set_title(f"Intensity\nSession: {session}", fontsize=int(textsize*1.3))
            ax.set_xlabel("Well Column", fontsize=int(textsize*1.2))
            ax.set_ylabel("Well Row", fontsize=int(textsize*1.2))
            plt.tight_layout()
            out_path = os.path.join(output_dir,
                                    f"{device_id}_intensity_heatmap_{session}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

        else:
            # split into two halves
            fig, axes = plt.subplots(1, 2,
                                     figsize=((len(col_labels))+10, len(row_labels)+1),
                                     constrained_layout=False)

            # define halves
            halves = [
                (range(0,6), cell_lines[0]),
                (range(6,12), cell_lines[1])
            ]
            for ax, (col_idx, cl_label) in zip(axes, halves):
                raw_sub = raw[:, col_idx]
                mn, mx = np.nanmin(raw_sub), np.nanmax(raw_sub)
                denom = (mx - mn) or 1.0
                norm_sub = (raw_sub - mn) / denom
                mask_sub = np.isnan(raw_sub)
                cols_sub = col_labels[col_idx.start:col_idx.stop]

                sns.heatmap(norm_sub, mask=mask_sub, annot=norm_sub, fmt=".3f",
                            cmap=cmap, vmin=0, vmax=1,
                            xticklabels=cols_sub, yticklabels=row_labels,
                            annot_kws={"fontsize":textsize},
                            cbar=show_cbar, ax=ax, square=True)
                ax.set_title(f"{device_id} {cl_label}\nSession: day {session}", fontsize=int(textsize*1.3))
                ax.set_xlabel("Col", fontsize=int(textsize*1.2))
                ax.set_ylabel("Row", fontsize=int(textsize*1.2))
                ax.tick_params(axis='both', labelsize=textsize)

                # custom cbar ticks
                if show_cbar:
                    cbar = ax.collections[0].colorbar

                    # 1) pick 5 evenly spaced ticks in normalized [0,1] space
                    ticks = np.linspace(0, 1, 5)
                    cbar.set_ticks(ticks)

                    cbar.ax.yaxis.set_major_formatter(mtick.FuncFormatter(short_formatter))
                    cbar.ax.tick_params(labelsize=textsize)
                    cbar.set_label("Intensity (×10⁶)", fontsize=int(textsize * 1.2))

            # save combined figure
            out_path = os.path.join(output_dir,
                                    f"{device_id}_intensity_heatmap_split_{session}.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)

    print(f"Saved heatmaps to {output_dir}")