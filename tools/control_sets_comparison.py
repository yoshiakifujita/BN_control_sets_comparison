#from cana.boolean_network import BooleanNetwork
import pandas as pd
import numpy as np
from cana.boolean_network import BooleanNetwork as BN
from pathlib import Path

from itertools import product
import re

import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

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

def visualize_structural_control_nodes_vs_ground_truth(str_reference_path, measure_type, nrows, ncols):
    """Visualize similarity (Jaccard or undershoot) between ground truth driver node sets and structural control sets (FVS, MDS, and SC) of interaction graphs and effective graphs across distinct edge thresholds for target models.
    
    Args:
        str_reference_path (str): Path to the directory where target Boolean network structural control set information are stored.
        measure_type (str)      : Specify similarity either Jaccard or undershoot
    """
    
    # Gather data (order by nomber of nodes)
    model_folders = gather_data(str_reference_path)

    
    # Set up 5x8 subplot grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(28, 16))
    axes = axes.flatten()

    # Set up plots colors
    fvs_color = "tab:blue"
    mds_color = "tab:orange"
    sc_color = "tab:green"
    
    legend_lines = None
    
    # Loop over each model directory with index
    model_title_mapping=[]

    # Iterate visualizations for each model
    for i, folder in enumerate(model_folders[:40]):
        model_name = folder.name
        display_title = f"Model {i + 1}"

        # Load Boolean network model
        input_cnet = "../datasets/targets/" + model_name + ".txt"
        bn = BN.from_file(input_cnet, type="cnet")
        IG = bn.structural_graph()

        # Get distinct edge effectiveness threshold
        csv_file = folder / "edge_effectiveness.csv"
        edge_counts = pd.read_csv(csv_file)

        # Compute similarity score specified by measure_type argument, either Jaccard or undershoot
        IG_Score, similarity_scores = (
            compare_ground_truth_structural_control_sets_iterate(
                bn=bn,
                f=folder,
                edge_counts=edge_counts,
                measure_type=measure_type,
            )
        )

        # Extract raw threshold vectors from Jaccard scores
        raw_thresholds = [r[1] for r in similarity_scores]
        raw_fvs = [r[2]["mean"] for r in similarity_scores]
        raw_mds = [r[3]["mean"] for r in similarity_scores]
        raw_sc = [r[4]["mean"] for r in similarity_scores]
    
        number_of_measurements = str(len(edge_counts)-1)

        model_title_mapping.append(
            {
                "display_title": display_title,
                "model_name": model_name,
                "number_of_nodes": IG.number_of_nodes(),
                "number_of_edges": IG.number_of_edges(),
                "number_of_measurements": number_of_measurements,
            }
        )

        # Apply downsampling capped at max_points=15
        thresholds, fvs_mean = downsample_multi_series(raw_thresholds, raw_fvs, max_points=15)
        _, mds_mean = downsample_multi_series(raw_thresholds, raw_mds, max_points=15)
        _, sc_mean = downsample_multi_series(raw_thresholds, raw_sc, max_points=15)
    
        display_title = display_title + f" #nodes {IG.number_of_nodes()}"

        ig_x = -0.05
        ig_fvs = IG_Score[2]["mean"]
        ig_mds = IG_Score[3]["mean"]
        ig_sc = IG_Score[4]["mean"]
    
        ax = axes[i]
    
        # Compute Smoothed Curves (Uniform x-grid) ---
        x_fvs_smooth, y_fvs_smooth = get_smooth_curve(thresholds, fvs_mean)
        x_mds_smooth, y_mds_smooth = get_smooth_curve(thresholds, mds_mean)
        x_sc_smooth, y_sc_smooth = get_smooth_curve(thresholds, sc_mean)
    
        # Plot Smooth Lines + Raw Markers ---
        l1 = ax.plot(x_fvs_smooth, y_fvs_smooth, color=fvs_color, lw=1.8)[0]
        l2 = ax.plot(x_mds_smooth, y_mds_smooth, color=mds_color, lw=1.8)[0]
        l3 = ax.plot(x_sc_smooth, y_sc_smooth, color=sc_color, lw=1.8)[0]
    
        # Overlay raw discrete observation points as markers
        ax.plot(
            thresholds,
            fvs_mean,
            color=fvs_color,
            marker="s",
            ls="None",
            ms=4,
            alpha=0.7,
        )
        ax.plot(
            thresholds,
            mds_mean,
            color=mds_color,
            marker="^",
            ls="None",
            ms=4,
            alpha=0.7,
        )
        ax.plot(
            thresholds,
            sc_mean,
            color=sc_color,
            marker="d",
            ls="None",
            ms=4,
            alpha=0.7,
        )
    
        # Plot initial graph (IG) baseline points ---
        ax.plot(
            ig_x,
            ig_fvs,
            color=fvs_color,
            marker="s",
            ms=6,
            mec="black",
            mew=0.8,
            ls="None",
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
            )
    
        ax.set_ylim(-0.05, 1.05)
    
        # Y-label on the left-most column
        if i % 8 == 0:
            ax.set_ylabel(f"Avg {measure_type}", fontsize=10)
    
        ax.tick_params(axis="y", labelsize=8)
    
        if legend_lines is None:
            legend_lines = [l1, l2, l3]
    
        # Subplot Title & X-ticks ---
        ax.set_title(display_title, fontsize=9, pad=4)
    
        # X-Ticks on all subplots (showing raw threshold steps)
        xticks = [ig_x] + list(thresholds)
        xticklabels = ["IG"] + [f"{t:.2f}" for t in thresholds]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=45, fontsize=7)

    # Global Figure Legend & Layout ---
    fig.legend(
        legend_lines,
        ["FVS", "MDS", "SC"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        fontsize=12,
        frameon=True,
    )
    
    plt.subplots_adjust(
        top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=0.4, wspace=0.35
    )
    
    plt.savefig(
        "all_40_networks_analysis_jaccard_smoothed_new.png",
        dpi=300,
        bbox_inches="tight",
    )

    return plt

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

def compare_list_of_lists(list1, list2, measure_type):
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
    if measure_type == 'Jaccard':
        scores = [
            jaccard_similarity(a, b)
            for a, b in product(list1, list2)
        ]
    elif measure_type == 'Overshoot':
        scores = [
            overshoot(a, b)
            for a, b in product(list1, list2)
        ]
    elif measure_type == 'Undershoot':
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

def compare_ground_truth_structural_control_sets_iterate(bn, f, edge_counts, measure_type):
    """Compute similarity scores in either Jaccard, overshoot, or undershoot between structural control sets (FVS, MDS, and SC) and ground truth driver node sets, of interaction graphs and effective graphs across distinct edge thresholds for target models.
    
    Args:
        bn (cana Boolean network object): Target model Boolean network object.
        f  (str)                        : Path to the directory where target model's structural control sets are stored.
        edge_counts (pandas dataframe)  : Target model's distinct edge effectiveness are stored.
        measure_type (str)              : Similarity type, either Jaccard, Overshoot, or Undershoot is accepted.
    Returns:
        IG_Score (list)       : Return similarity score of the interaction graph.
        similarity_scores (list) : Return similarity score of the effective graphs with distinct edge effectiveness thresholds.
    """
    
    # Compute similarity score of interaction graph
    EG=bn.effective_graph()
    input_csv=f / 'st_control_ig.csv'
    
    df_ig=pd.read_csv(input_csv)
    driver_sets, fvs_sets, mds_sets, sc_sets = summarize_control_sets(df_ig=df_ig)
    
    ig_fvs_scores = compare_list_of_lists(driver_sets, fvs_sets, measure_type)
    ig_mds_scores = compare_list_of_lists(driver_sets, mds_sets, measure_type)
    ig_sc_scores = compare_list_of_lists(driver_sets, sc_sets, measure_type)
    
    IG_Score = [EG.number_of_edges(), 0.0, ig_fvs_scores, ig_mds_scores, ig_sc_scores]
    
    # Compute similarity score of effective graph with edge effectiveness threshold 0.0
    
    similarity_scores = []
    threshold=0.0
    EG=bn.effective_graph(threshold=threshold)
    
    input_csv=f / 'st_control_eg_00.csv'
    df_eg=pd.read_csv(input_csv)
    driver_sets, fvs_sets, mds_sets, sc_sets = summarize_control_sets(df_ig=df_eg)
    
    eg_fvs_scores = compare_list_of_lists(driver_sets, fvs_sets, measure_type)
    eg_mds_scores = compare_list_of_lists(driver_sets, mds_sets, measure_type)
    eg_sc_scores = compare_list_of_lists(driver_sets, sc_sets, measure_type)
    
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
    
        eg_fvs_scores = compare_list_of_lists(driver_sets, fvs_sets, measure_type)
        eg_mds_scores = compare_list_of_lists(driver_sets, mds_sets, measure_type)
        eg_sc_scores = compare_list_of_lists(driver_sets, sc_sets, measure_type)
        
        tmp_list = [EG.number_of_edges(), threshold, eg_fvs_scores, eg_mds_scores, eg_sc_scores]
        similarity_scores.append(tmp_list)
        n=n+1

    return IG_Score, similarity_scores