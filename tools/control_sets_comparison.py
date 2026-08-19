#from cana.boolean_network import BooleanNetwork
import pandas as pd
import numpy as np
from cana.boolean_network import BooleanNetwork as BN
from pathlib import Path

from itertools import product
import re

import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

from scipy import stats
import textwrap
from sklearn.metrics import roc_curve, auc, precision_score, accuracy_score, recall_score

def actionability_per_node_boxplot(df_actionability_all, analysis_level):

    if analysis_level=="Edge":
        target_column='edge_actionability'
        analysis_title='Edge Level Analysis'
    elif analysis_level=="Node":
        target_column='node_actionability'
        analysis_title='Node Level Analysis'        

    # 1. Separate the groups
    drivers = df_actionability_all[df_actionability_all['driver_node'] == 'driver'][target_column]
    non_drivers = df_actionability_all[df_actionability_all['driver_node'] == 'non-driver'][target_column]
    
    # 2. Compute Statistics
    # We use Mann-Whitney U as it's robust for biological viability data
    u_stat, p_val_u = stats.mannwhitneyu(drivers, non_drivers, alternative='two-sided')
    
    # 3. Set up the figure
    plt.figure(figsize=(4.2, 6))
    
    # 4. Create the labels with Sample Sizes (n=)
    n_drivers = len(drivers) / 2
    n_non_drivers = len(non_drivers) / 2
    labels = [f'Driver \n(n={n_drivers})', f'Non-Driver \n(n={n_non_drivers})']
    
    # 5. Plotting
    data_to_plot = [drivers, non_drivers]
    #bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, notch=True, widths=0.4)
    
    bp = plt.boxplot(
        data_to_plot,
        tick_labels=labels,
        patch_artist=True,
        notch=True,
        widths=0.4,
        showmeans=True,
        medianprops=dict(color='black', linewidth=2.5),
        meanprops=dict(
            marker='D',
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=8
        ),
        boxprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5)
    )
    
    # 6. Aesthetics
    colors = ['crimson', 'royalblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # 7. Add the Statistics (U-stat and P-Value) to the plot
    y_max = max(df_actionability_all[target_column])
    y_pos = y_max * 1.05 
    
    # Format strings
    # Using integer for u_stat as it is a count of rank sums
    #u_text = f'$U = {u_stat:,.0f}$' 
    p_text = f'$p = {p_val_u:.2e}$' if p_val_u < 0.001 else f'$p = {p_val_u:.4f}$'
    
    # Combine texts with a newline
    # We place this at y_pos * 1.02 to ensure the bracket doesn't overlap the text
    #stats_label = f"{u_text}\n{p_text}"
    stats_label = f"{(p_text)}"
    
    # Draw the line
    plt.plot([1, 1, 2, 2], [y_pos*0.98, y_pos, y_pos, y_pos*0.98], lw=1.5, c='black')
    
    # Place the text
    plt.text(1.5, y_pos * 1.02, stats_label, ha='center', va='bottom', 
             fontsize=16, fontweight='bold', linespacing=1.5)
    
    # Adjust y-limit to ensure the double-line text isn't cut off
    plt.ylim(0, y_pos * 1.25)
    
    # --- MODIFICATIONS START HERE ---
    
    # 1. Set Y-limit with a negative margin (e.g., -0.05) to add space under 0
    # The top remains high enough to fit the stats
    plt.ylim(-0.05, y_pos * 1.2)
    
    # 2. Explicitly set Y-ticks to stop at 1.0
    # This prevents 1.2 or other high numbers from appearing on the axis
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14)
    
    # 8. Labels and Title
    plt.title(analysis_title, fontsize=18, pad=20)
    #plt.ylabel('Actionability', fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xticks(fontsize=18)
    
    # 9. Save and Show
    plt.tight_layout()
    #plt.savefig('edge_actionability_with_stats.png', dpi=400)
    plt.show()

def downsample_multi_series(x, y_data, max_points=15):
    """
    Downsamples x and y_data to at most max_points.
    y_data can be a single list/array or a list of lists/arrays.
    """
    x_arr = np.asarray(x)
    n_points = len(x_arr)
    
    # Handle empty or small series without downsampling
    if n_points <= max_points:
        return x, y_data
    
    # Generate evenly spaced indices
    float_indices = np.linspace(0, n_points - 1, max_points)
    indices = np.unique(np.round(float_indices).astype(int))
    
    x_down = x_arr[indices].tolist()
    
    # Check if y_data is a single 1D series (e.g., [0.1, 0.2, ...]) 
    # or a container of multiple series (e.g., [[...], [...]])
    if len(y_data) > 0 and not isinstance(y_data[0], (list, np.ndarray)):
        # Single 1D list/array passed
        y_down = np.asarray(y_data)[indices].tolist()
    else:
        # Multiple series passed as a list of lists/arrays
        y_down = [np.asarray(y)[indices].tolist() for y in y_data]
        
    return x_down, y_down

def get_smooth_curve(x_raw, y_raw, num_points=100):
    """Generates smooth curves using Monotone Cubic Spline (PCHIP).

    Guarantees curves never overshoot actual min/max data points.
    """
    x_raw = np.array(x_raw, dtype=float)
    y_raw = np.array(y_raw, dtype=float)

    # 1. Filter out NaNs
    valid_mask = ~np.isnan(y_raw) & ~np.isnan(x_raw)
    x_clean, y_clean = x_raw[valid_mask], y_raw[valid_mask]

    if len(x_clean) < 2:
        return x_clean, y_clean

    # 2. Group duplicate x-values safely
    df_temp = (
        pd.DataFrame({"x": x_clean, "y": y_clean})
        .groupby("x", as_index=False)
        .mean()
        .sort_values("x")
    )
    x_clean, y_clean = df_temp["x"].values, df_temp["y"].values

    if len(x_clean) < 2:
        return x_clean, y_clean

    # 3. Non-overshooting interpolation
    x_grid = np.linspace(x_clean.min(), x_clean.max(), num_points)

    if len(x_clean) >= 3:
        f_interp = PchipInterpolator(x_clean, y_clean)
        y_smooth = f_interp(x_grid)
    else:
        y_smooth = np.interp(x_grid, x_clean, y_clean)

    # Safety clamp to actual data range
    y_smooth = np.clip(y_smooth, y_clean.min(), y_clean.max())

    return x_grid, y_smooth

def get_target_bn_path (str_base_dir):
    # Resolve absolute path to avoid current working directory confusion
    base_dir = Path(str_base_dir).resolve()
    
    if base_dir.exists():
        all_items = list(base_dir.iterdir())
        print(f"Total items found (files + folders): {len(all_items)}")
    
        # Check if items are files instead of subdirectories
        dirs = [f for f in all_items if f.is_dir()]
        files = [f for f in all_items if f.is_file()]
    
        print(f"Subdirectories: {len(dirs)}")
        print(f"Files: {len(files)}")
    
        # Use directories if available; otherwise fall back to files or stem names
        if len(dirs) > 0:
            model_paths = sorted(dirs)[:40]
        else:
            print(
                "Notice: Items in target directory are files, not subdirectories."
            )
            model_paths = sorted(all_items)[:40]
    
        print(f"Final model_folders count: {len(model_paths)}")
    
    return model_paths

def gather_data(str_reference_path):
    base_dir = Path(str_reference_path)
    
    # Get all subdirectories (models) and sort them for consistent grid order
    # Filter out non-directories or system files
    model_folders = sorted([f for f in base_dir.iterdir() if f.is_dir()])
    
    # Optional: Ensure you only take up to 40 models
    #model_folders = model_folders[:40]
    
    print(f"Found {len(model_folders)} model directories.")

    # Order models by number of nodes
    model_records = []
    
    # Collect network metadata for all 40 models
    for i, folder in enumerate(model_folders[:40]):
        model_name = folder.name
    
        input_cnet = "../datasets/targets/" + model_name + ".txt"
        bn = BN.from_file(input_cnet, type="cnet")
        IG = bn.structural_graph()
    
        nNodes = IG.number_of_nodes()
        nEdges = IG.number_of_edges()
    
        model_records.append(
            {
                "original_index": i + 1,
                "folder_path": folder,
                "model_name": model_name,
                "nNodes": nNodes,
                "nEdges": nEdges,
            }
        )
    
    # Create DataFrame
    model_df = pd.DataFrame(model_records)
    
    # 2. Map model_name to nNodes for O(1) fast lookup
    nodes_dict = dict(zip(model_df["model_name"], model_df["nNodes"]))
    
    # 3. Sort model_folders by nNodes (falling back to float('inf') if a model isn't found in the CSV)
    model_folders = sorted(
        model_folders, key=lambda folder: nodes_dict.get(folder.name, float("inf"))
    )

    return model_folders

def structural_control_nodes_vs_ground_truth(
    str_reference_path, similarity_type, summary_type, nrows, ncols, max_observations
):
    """Return metrics (Jaccard, Undershoot, or Overshoot) of Interaction graph for debug

    driver node sets and structural control sets (FVS, MDS, and SC) across distinct edge thresholds.

    Args:
        str_reference_path (str): Path to directory where Boolean network files are stored.
        similarity_type (str)   : Metric type - "Jaccard", "Undershoot", or "Overshoot"
        summary_type (str)      : Aggregation method - "mean", "max", "min", "max-mean", or "min-mean"
        nrows (int)             : Number of subplot rows
        ncols (int)             : Number of subplot columns
        max_observations (int)  : Maximum observations of distinct edge effectiveness
    """

    # Gather model folders ordered by node count
    model_folders = gather_data(str_reference_path)

    model_title_mapping = []
    ig_score_records = []

    # Iterate across models
    for i, folder in enumerate(model_folders):
        model_name = folder.name
        display_title = f"Model {i + 1}"

        # Load Boolean network and network structure
        input_cnet = f"../datasets/targets/{model_name}.txt"
        bn = BN.from_file(input_cnet, type="cnet")
        IG = bn.structural_graph()

        csv_file = folder / "edge_effectiveness.csv"
        edge_counts = pd.read_csv(csv_file)

        # Compute metric score across thresholds
        IG_Score, similarity_scores = (
            compare_ground_truth_structural_control_sets_iterate(
                bn=bn,
                f=folder,
                edge_counts=edge_counts,
                similarity_type=similarity_type,
                summary_type=summary_type
            )
        )

        ig_score_records.append({
                    "model_index": i + 1,
                    "model_name": model_name,
                    "number_of_nodes": IG.number_of_nodes(),
                    "number_of_edges": IG.number_of_edges(),
                    "raw_ig_score": IG_Score
                })

    return ig_score_records

def visualize_structural_control_nodes_vs_ground_truth(
    str_reference_path, similarity_type, summary_type, nrows, ncols, max_observations
):
    """Visualize metric curves (Jaccard, Undershoot, or Overshoot) between ground truth
    driver node sets and structural control sets (FVS, MDS, and SC) across distinct edge thresholds.
    """

    # Helper to safely extract metric score regardless of summary_type format
    def get_score(score_dict, summary, default=0.0):
        if not isinstance(score_dict, dict):
            return default

        # 1. Try 'summary_score' first (used in max-mean, min-mean, etc.)
        val = score_dict.get("summary_score")
        if val is not None:
            return val

        # 2. Try explicit summary key (used in 'max', 'min', 'mean')
        val = score_dict.get(summary)
        if val is not None:
            return val

        # 3. Fallback to 'mean' or provided default
        val = score_dict.get("mean")
        return val if val is not None else default

    # Gather model folders ordered by node count
    model_folders = gather_data(str_reference_path)

    total_plots = nrows * ncols
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 3.5, nrows * 3.2)
    )
    axes = axes.flatten()

    # Define color scheme
    fvs_color = "tab:blue"
    mds_color = "tab:orange"
    sc_color = "tab:green"

    legend_lines = None
    model_title_mapping = []
    ig_score_records = []

    # Iterate across models
    for i, folder in enumerate(model_folders[:total_plots]):
        model_name = folder.name
        display_title = f"Model {i + 1}"

        # Load Boolean network and network structure
        input_cnet = f"../datasets/targets/{model_name}.txt"
        bn = BN.from_file(input_cnet, type="cnet")
        IG = bn.structural_graph()

        csv_file = folder / "edge_effectiveness.csv"
        edge_counts = pd.read_csv(csv_file)

        # Compute metric score across thresholds
        IG_Score, similarity_scores = (
            compare_ground_truth_structural_control_sets_iterate(
                bn=bn,
                f=folder,
                edge_counts=edge_counts,
                similarity_type=similarity_type,
                summary_type=summary_type
            )
        )

        # Extract raw threshold vectors using unified get_score
        raw_thresholds = [r[1] for r in similarity_scores]
        raw_fvs = [get_score(r[2], summary_type) for r in similarity_scores]
        raw_mds = [get_score(r[3], summary_type) for r in similarity_scores]
        raw_sc = [get_score(r[4], summary_type) for r in similarity_scores]

        ig_score_records.append({
                    "model_index": i + 1,
                    "model_name": model_name,
                    "number_of_nodes": IG.number_of_nodes(),
                    "number_of_edges": IG.number_of_edges(),
                    "raw_ig_score": IG_Score
                })

        # Log metadata
        model_title_mapping.append(
            {
                "display_title": display_title,
                "model_name": model_name,
                "number_of_nodes": IG.number_of_nodes(),
                "number_of_edges": IG.number_of_edges(),
                "number_of_measurements": str(len(edge_counts) - 1),
            }
        )

        # Synchronized downsampling capped at max_observations
        thresholds, [fvs_pts, mds_pts, sc_pts] = downsample_multi_series(
            raw_thresholds,
            [raw_fvs, raw_mds, raw_sc],
            max_points=max_observations,
        )

        ig_x = -0.05
        ig_fvs = get_score(IG_Score[2], summary_type)
        ig_mds = get_score(IG_Score[3], summary_type)
        ig_sc = get_score(IG_Score[4], summary_type)
        ax = axes[i]

        # Compute smoothed curve paths
        x_fvs_smooth, y_fvs_smooth = get_smooth_curve(thresholds, fvs_pts)
        x_mds_smooth, y_mds_smooth = get_smooth_curve(thresholds, mds_pts)
        x_sc_smooth, y_sc_smooth = get_smooth_curve(thresholds, sc_pts)

        # Calculate Y-axis bounds
        model_y_max = max(
            np.nanmax(y_fvs_smooth) if len(y_fvs_smooth) else 0,
            np.nanmax(y_mds_smooth) if len(y_mds_smooth) else 0,
            np.nanmax(y_sc_smooth) if len(y_sc_smooth) else 0,
            ig_fvs or 0,
            ig_mds or 0,
            ig_sc or 0 if ig_sc is not None else 0,
        )

        is_overshoot = similarity_type.lower() == "overshoot"
        y_upper_limit = (
            max(1.05, model_y_max * 1.08) if is_overshoot else 1.05
        )

        # Overshoot visual cues (horizontal reference line, shading, and peak text)
        if is_overshoot and model_y_max > 1.0:
            ax.axhline(
                1.0,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
                zorder=1,
            )
            ax.axhspan(1.0, y_upper_limit, color="red", alpha=0.04, zorder=0)
            ax.text(
                0.95,
                0.92,
                f"Peak: {model_y_max:.1f}",
                transform=ax.transAxes,
                fontsize=7,
                color="crimson",
                fontweight="bold",
                ha="right",
                va="top",
            )

        # Plot smooth curves
        l1 = ax.plot(
            x_fvs_smooth, y_fvs_smooth, color=fvs_color, lw=1.8, zorder=2
        )[0]
        l2 = ax.plot(
            x_mds_smooth, y_mds_smooth, color=mds_color, lw=1.8, zorder=2
        )[0]
        l3 = ax.plot(
            x_sc_smooth, y_sc_smooth, color=sc_color, lw=1.8, zorder=2
        )[0]

        # Discrete observation markers
        ax.plot(
            thresholds,
            fvs_pts,
            color=fvs_color,
            marker="s",
            ls="None",
            ms=4,
            alpha=0.85,
            zorder=3,
        )
        ax.plot(
            thresholds,
            mds_pts,
            color=mds_color,
            marker="^",
            ls="None",
            ms=4,
            alpha=0.85,
            zorder=3,
        )
        ax.plot(
            thresholds,
            sc_pts,
            color=sc_color,
            marker="d",
            ls="None",
            ms=4,
            alpha=0.85,
            zorder=3,
        )

        # IG baseline markers
        ax.plot(
            ig_x,
            ig_fvs,
            color=fvs_color,
            marker="s",
            ms=6,
            mec="black",
            mew=0.8,
            ls="None",
            zorder=3,
        )
        ax.plot(
            ig_x,
            ig_mds,
            color=mds_color,
            marker="^",
            ms=6,
            mec="black",
            mew=0.8,
            ls="None",
            zorder=3,
        )
        if ig_sc is not None:
            ax.plot(
                ig_x,
                ig_sc,
                color=sc_color,
                marker="d",
                ms=6,
                mec="black",
                mew=0.8,
                ls="None",
                zorder=3,
            )

        ax.set_ylim(-0.05, y_upper_limit)

        # Set Y-axis labels on left-most column
        if i % ncols == 0:
            ax.set_ylabel(
                f"{summary_type.capitalize()} {similarity_type}", fontsize=10
            )

        ax.tick_params(axis="y", labelsize=8)

        if legend_lines is None:
            legend_lines = [l1, l2, l3]

        # Subplot Title & X-ticks
        ax.set_title(
            f"{display_title} #nodes {IG.number_of_nodes()}",
            fontsize=9,
            pad=4,
        )
        xticks = [ig_x] + list(thresholds)
        xticklabels = ["IG"] + [f"{t:.2f}" for t in thresholds]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, fontsize=7)

    # Clean up empty subplots if nrows * ncols > len(model_folders)
    for j in range(i + 1, total_plots):
        fig.delaxes(axes[j])

    # 1. Combined Overall Title & Legend Text in Suptitle
    # Or format as: f"{similarity_type.capitalize()} ({summary_type.capitalize()})   —   Legend:  ■ FVS   ▲ MDS   ◆ SC"
    combined_title = f"{similarity_type.capitalize()} ({summary_type.capitalize()})"
    fig.suptitle(combined_title, fontsize=15, fontweight="bold", y=0.98, x=0.25, ha="left")

    # 2. Global Legend placed on the right side at the exact same vertical position (y=0.98)
    fig.legend(
        legend_lines,
        ["FVS", "MDS", "SC"],
        loc="upper right",
        bbox_to_anchor=(0.95, 0.995),
        ncol=3,
        fontsize=11,
        frameon=True,
    )

    # 3. Adjust top margin to accommodate the single top header row
    plt.subplots_adjust(
        top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=0.45, wspace=0.35
    )

    out_file = f"all_networks_analysis_{similarity_type.lower()}_{summary_type.lower()}.png"
    
    plt.savefig(out_file, dpi=300, bbox_inches="tight")

    return plt, ig_score_records, model_title_mapping

def summarize_control_sets(df_ig):

    driver_sets = []
    
    # Select only columns like is_driver_set_1, is_driver_set_2, ...
    driver_cols = [
        col for col in df_ig.columns
        if re.fullmatch(r"is_driver_set_\d+", col)
    ]
    
    # Sort them numerically (optional, but recommended)
    driver_cols = sorted(driver_cols, key=lambda x: int(x.split("_")[-1]))
    
    # Create the list of lists
    for col in driver_cols:
        drivers = df_ig.loc[df_ig[col] == 1, "node_id"].tolist()
        driver_sets.append(drivers)
    
    #print(driver_sets)
    
    fvs_sets = []
    
    # Select only columns like is_FVS_1, is_FVS_2, ...
    fvs_cols = [
        col for col in df_ig.columns
        if re.fullmatch(r"is_FVS_\d+", col)
    ]
    
    # Sort them numerically
    fvs_cols = sorted(fvs_cols, key=lambda x: int(x.split("_")[-1]))
    
    # Create the list of lists
    for col in fvs_cols:
        fvs = df_ig.loc[df_ig[col] == 1, "node_id"].tolist()
        fvs_sets.append(fvs)
    
    #print(fvs_sets)
    
    mds_sets = []
    
    # Select only columns like is_FVS_1, is_FVS_2, ...
    mds_cols = [
        col for col in df_ig.columns
        if re.fullmatch(r"is_MDS_\d+", col)
    ]
    
    # Sort them numerically
    mds_cols = sorted(mds_cols, key=lambda x: int(x.split("_")[-1]))
    
    # Create the list of lists
    for col in mds_cols:
        mds = df_ig.loc[df_ig[col] == 1, "node_id"].tolist()
        mds_sets.append(mds)
    
    #print(mds_sets)
    
    sc_sets = []
    
    # Select only columns like is_FVS_1, is_FVS_2, ...
    sc_cols = [
        col for col in df_ig.columns
        if re.fullmatch(r"is_SC_\d+", col)
    ]
    
    # Sort them numerically
    sc_cols = sorted(sc_cols, key=lambda x: int(x.split("_")[-1]))
    
    # Create the list of lists
    for col in sc_cols:
        sc = df_ig.loc[df_ig[col] == 1, "node_id"].tolist()
        sc_sets.append(sc)
    
    #print(sc_sets)
    return driver_sets, fvs_sets, mds_sets, sc_sets

def jaccard_similarity(g, p):
    """Computes the Jaccard similarity (False Positives / |Ground Truth|).
    Inputs:
        g: iterable of ground truth elements
        p: iterable of predicted elements
    Returns:
        float: accard similarity
    """    
    set_g = set(g)
    set_p = set(p)

    union = set_g | set_p
    if len(union) == 0:
        return None  # shouldn't occur after the check below

    return len(set_g & set_p) / len(union)

def overshoot(g, p):
    """Computes the Overshoot Rate (False Positives / |Ground Truth|).

    Inputs:
        g: iterable of ground truth elements
        p: iterable of predicted elements
    Returns:
        float: Overshoot rate normalized by ground truth size
    """
    set_g = set(g)
    set_p = set(p)

    if not set_g:
        return 0.0

    return len(set_p - set_g) / len(set_g)


def undershoot(g, p):
    """Computes the Undershoot Rate (False Negatives / |Ground Truth|).

    Inputs:
        g: iterable of ground truth elements
        p: iterable of predicted elements
    Returns:
        float: Undershoot rate normalized by ground truth size
    """
    set_g = set(g)
    set_p = set(p)

    if not set_g:
        return 0.0

    return len(set_g - set_p) / len(set_g)

def compare_list_of_lists(list1, list2, similarity_type):
    """
    Compute pairwise similarities between every sublist in list1
    and every sublist in list2.

    Returns similarity statistics, mean, maximum, minimum, and all similarity scores.
    """

    # Remove empty sublists
    list1 = [x for x in list1 if len(x) > 0]
    list2 = [x for x in list2 if len(x) > 0]

    # If either becomes empty, return None
    if len(list1) == 0 or len(list2) == 0:
        return {
            "max": None,
            "min": None,
            "mean": None,
            "all_scores": []
        }
    if similarity_type == 'Jaccard':
        scores = [
            jaccard_similarity(a, b)
            for a, b in product(list1, list2)
        ]
    elif similarity_type == 'Overshoot':
        scores = [
            overshoot(a, b)
            for a, b in product(list1, list2)
        ]
    elif similarity_type == 'Undershoot':
        scores = [
            undershoot(a, b)
            for a, b in product(list1, list2)
        ]
        
    return {
        "max": max(scores),
        "min": min(scores),
        "mean": sum(scores) / len(scores),
        "all_scores": scores
    }

from itertools import product

def two_steps_compare_list_of_lists(list1, list2, similarity_type, summary_type):
    """
    Computes pairwise similarities between ground truth sublists (list1) 
    and prediction sublists (list2), summarizing across predictions per ground truth.

    :param list1: List of ground truth sublists [g1, g2, g3]
    :param list2: List of prediction sublists [p1, p2]
    :param similarity_type: 'Jaccard', 'Overshoot', or 'Undershoot'
    :param summary_type: 'max-mean' (mean of GT maxes) or 'min-mean' (mean of GT mins)
    :return: Dictionary containing the aggregated final score and per-GT scores
    """
    # Remove empty sublists
    list1 = [x for x in list1 if len(x) > 0]
    list2 = [x for x in list2 if len(x) > 0]

    # Return None if either input list becomes empty
    if len(list1) == 0 or len(list2) == 0:
        return {
            "summary_score": None,
            "gt_scores": [],
            "summary_type": summary_type
        }

    # Select the metric function
    sim_funcs = {
        'Jaccard': jaccard_similarity,
        'Overshoot': overshoot,
        'Undershoot': undershoot
    }
    
    if similarity_type not in sim_funcs:
        raise ValueError(f"Unsupported similarity_type: {similarity_type}")
        
    metric_func = sim_funcs[similarity_type]

    # Step 1: For each GT (g_i), collect similarity scores against all predictions (p_j)
    gt_scores = []
    
    for g in list1:
        # Pairwise scores for current ground truth across all predictions
        preds_for_g = [metric_func(g, p) for p in list2]
        
        # Aggregate across predictions for this GT based on summary_type
        if summary_type == "max-mean":
            gt_scores.append(max(preds_for_g))
        elif summary_type == "min-mean":
            gt_scores.append(min(preds_for_g))
        else:
            raise ValueError(f"Unsupported summary_type: {summary_type}. Use 'max-mean' or 'min-mean'.")

    # Step 2: Compute mean across all ground truths
    overall_mean = sum(gt_scores) / len(gt_scores)

    return {
        "summary_score": overall_mean,
        "gt_scores": gt_scores,          # The max (or min) score computed for each GT
        "summary_type": summary_type
    }

def compare_ground_truth_structural_control_sets(bn, f, edge_counts):

    # Interaction graph
    EG=bn.effective_graph()
    
    input_csv=f + 'st_control_ig.csv'
    df_ig=pd.read_csv(input_csv)
    driver_sets, fvs_sets, mds_sets, sc_sets = summarize_control_sets(df_ig=df_ig)
    
    ig_fvs_scores = compare_list_of_lists(driver_sets, fvs_sets)
    ig_mds_scores = compare_list_of_lists(driver_sets, mds_sets)
    ig_sc_scores = compare_list_of_lists(driver_sets, sc_sets)
    
    IG_Score = [EG.number_of_edges(), 0.0, ig_fvs_scores, ig_mds_scores, ig_sc_scores]
    
    # Effective graph threshold 0.0
    
    jaccard_scores = []
    threshold=0.0
    EG=bn.effective_graph(threshold=threshold)
    
    # Effective graph, threshold 0.0
    input_csv=f + 'st_control_eg_00.csv'
    df_eg=pd.read_csv(input_csv)
    driver_sets, fvs_sets, mds_sets, sc_sets = summarize_control_sets(df_ig=df_eg)
    
    eg_fvs_scores = compare_list_of_lists(driver_sets, fvs_sets)
    eg_mds_scores = compare_list_of_lists(driver_sets, mds_sets)
    eg_sc_scores = compare_list_of_lists(driver_sets, sc_sets)
    
    tmp_list = [EG.number_of_edges(),threshold, eg_fvs_scores, eg_mds_scores, eg_sc_scores]
    
    jaccard_scores.append(tmp_list)

    # Effective graphs with distinct threshold
    n=1
    for threshold, count in zip(edge_counts["Edge effectiveness"], edge_counts["Count"]):
        if threshold==1.000: break
        threshold_dis=threshold+0.001
        EG=bn.effective_graph(threshold=threshold_dis)
        if EG.number_of_edges()<1.0: break
        input_csv = f"{f}st_control_eg_{n:02d}.csv"
        df_eg=pd.read_csv(input_csv)
        driver_sets, fvs_sets, mds_sets, sc_sets = summarize_control_sets(df_ig=df_eg)
    
        eg_fvs_scores = compare_list_of_lists(driver_sets, fvs_sets)
        eg_mds_scores = compare_list_of_lists(driver_sets, mds_sets)
        eg_sc_scores = compare_list_of_lists(driver_sets, sc_sets)
        
        tmp_list = [EG.number_of_edges(), threshold, eg_fvs_scores, eg_mds_scores, eg_sc_scores]
        jaccard_scores.append(tmp_list)
        n=n+1

    return IG_Score, jaccard_scores

def compare_ground_truth_structural_control_sets_iterate(bn, f, edge_counts, similarity_type, summary_type):
    """Compute similarity scores in either Jaccard, overshoot, or undershoot between structural control sets (FVS, MDS, and SC) and ground truth driver node sets, of interaction graphs and effective graphs across distinct edge thresholds for target models.
    
    Args:
        bn (cana Boolean network object): Target model Boolean network object.
        f  (str)                        : Path to the directory where target model's structural control sets are stored.
        edge_counts (pandas dataframe)  : Target model's distinct edge effectiveness are stored.
        similarity_type (str)           : Similarity type, either Jaccard, Overshoot, or Undershoot is accepted.
    Returns:
        IG_Score (list)       : Return similarity score of the interaction graph.
        similarity_scores (list) : Return similarity score of the effective graphs with distinct edge effectiveness thresholds.
    """
    
    # Compute similarity score of interaction graph
    EG=bn.effective_graph()
    input_csv=f / 'st_control_ig.csv'
    
    df_ig=pd.read_csv(input_csv)
    driver_sets, fvs_sets, mds_sets, sc_sets = summarize_control_sets(df_ig=df_ig)

    ig_fvs_scores, ig_mds_scores, ig_sc_scores = compute_similarity_scores(driver_sets, fvs_sets, mds_sets, sc_sets, similarity_type, summary_type)
    
    #ig_fvs_scores = compare_list_of_lists(driver_sets, fvs_sets, similarity_type)
    #ig_mds_scores = compare_list_of_lists(driver_sets, mds_sets, similarity_type)
    #ig_sc_scores = compare_list_of_lists(driver_sets, sc_sets, similarity_type)
    
    IG_Score = [EG.number_of_edges(), 0.0, ig_fvs_scores, ig_mds_scores, ig_sc_scores]
    
    # Compute similarity score of effective graph with edge effectiveness threshold 0.0
    
    similarity_scores = []
    threshold=0.0
    EG=bn.effective_graph(threshold=threshold)
    
    input_csv=f / 'st_control_eg_00.csv'
    df_eg=pd.read_csv(input_csv)
    driver_sets, fvs_sets, mds_sets, sc_sets = summarize_control_sets(df_ig=df_eg)

    eg_fvs_scores, eg_mds_scores, eg_sc_scores = compute_similarity_scores(driver_sets, fvs_sets, mds_sets, sc_sets, similarity_type, summary_type)
    
    #eg_fvs_scores = compare_list_of_lists(driver_sets, fvs_sets, similarity_type)
    #eg_mds_scores = compare_list_of_lists(driver_sets, mds_sets, similarity_type)
    #eg_sc_scores = compare_list_of_lists(driver_sets, sc_sets, similarity_type)
    
    tmp_list = [EG.number_of_edges(),threshold, eg_fvs_scores, eg_mds_scores, eg_sc_scores]
    
    similarity_scores.append(tmp_list)

    # Compute similarity score of effective graphs with distinct edge effectiveness iteratively
    n=1
    for threshold, count in zip(edge_counts["Edge effectiveness"], edge_counts["Count"]):
        if threshold==1.000: break
        threshold_dis=threshold+0.001
        EG=bn.effective_graph(threshold=threshold_dis)
        if EG.number_of_edges()<1.0: break

        input_csv = f / f"st_control_eg_{n:02d}.csv"
        df_eg=pd.read_csv(input_csv)
        driver_sets, fvs_sets, mds_sets, sc_sets = summarize_control_sets(df_ig=df_eg)

        eg_fvs_scores, eg_mds_scores, eg_sc_scores = compute_similarity_scores(driver_sets, fvs_sets, mds_sets, sc_sets, similarity_type, summary_type)

        #eg_fvs_scores = compare_list_of_lists(driver_sets, fvs_sets, similarity_type)
        #eg_mds_scores = compare_list_of_lists(driver_sets, mds_sets, similarity_type)
        #eg_sc_scores = compare_list_of_lists(driver_sets, sc_sets, similarity_type)
        
        tmp_list = [EG.number_of_edges(), threshold, eg_fvs_scores, eg_mds_scores, eg_sc_scores]
        similarity_scores.append(tmp_list)
        n=n+1

    return IG_Score, similarity_scores

def compute_similarity_scores(driver_sets, fvs_sets, mds_sets, sc_sets, similarity_type, summary_type):
    if summary_type == "max-mean" or summary_type == "min-mean":
        fvs_scores = two_steps_compare_list_of_lists(driver_sets, fvs_sets, similarity_type, summary_type)
        mds_scores = two_steps_compare_list_of_lists(driver_sets, mds_sets, similarity_type, summary_type)
        sc_scores = two_steps_compare_list_of_lists(driver_sets, sc_sets, similarity_type, summary_type)
    else:
        fvs_scores = compare_list_of_lists(driver_sets, fvs_sets, similarity_type)
        mds_scores = compare_list_of_lists(driver_sets, mds_sets, similarity_type)
        sc_scores = compare_list_of_lists(driver_sets, sc_sets, similarity_type)

    return fvs_scores, mds_scores, sc_scores