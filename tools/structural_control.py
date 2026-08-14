#from cana.boolean_network import BooleanNetwork
import pandas as pd
import numpy as np
import itertools
from cana.boolean_network import BooleanNetwork as BN
from pathlib import Path

from cana.control.fvs import fvs_bruteforce
from cana.control.mds import mds
from cana.control.sc import sc

def get_target_bn_path(str_base_dir):
    # Resolve absolute path
    base_dir = Path(str_base_dir).resolve()

    model_paths = []

    if not base_dir.exists():
        print(f"Directory does not exist: {base_dir}")
        return model_paths

    # Get only .txt files
    model_paths = sorted(
        [f for f in base_dir.iterdir()
         if f.is_file() and f.suffix.lower() == ".txt"]
    )[:40]

    print(f"Target .txt files found: {len(model_paths)}")
    print(f"Final model_paths count: {len(model_paths)}")

    return model_paths

def get_distinct_edge_effectiveness(bn):
    EG = bn.effective_graph()
    nodes=bn.nodes
    edge_data = [(nodes[u].name, nodes[v].name, EG[u][v]['weight']) for u, v in EG.edges()]
    
    df = pd.DataFrame(edge_data, columns=['Source', 'Target', 'Edge effectiveness'])
    
    # Truncate to 3 decimal places (no rounding)
    df["Edge effectiveness (truncated)"] = (
        np.trunc(df["Edge effectiveness"] * 1000) / 1000
    )
    
    # Count occurrences of each rounded value
    edge_counts = (
        df["Edge effectiveness (truncated)"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    
    # Rename columns
    edge_counts.columns = ["Edge effectiveness", "Count"]
    
    # Sort by Edge effectiveness (ascending)
    edge_counts = edge_counts.sort_values(
        by="Edge effectiveness",
        ascending=True
    ).reset_index(drop=True)
    
    return edge_counts

def compute_structural_control_nodes(str_base_dir, str_summary_dir, str_store_dir):
    """Compute structural control sets (FVS, MDS, and SC) of interaction graphs and effective graphs across distinct edge thresholds for target models.
    
    Args:
        str_base_dir (str):    Path to the directory where target Boolean network model text files are stored.
        str_summary_dir (str): Path to the directory where information of target Boolean network modes are stored.
        str_store_dir (str):   Path to the directory where summary results will be stored.
    """

    # Get target model paths
    model_paths=get_target_bn_path (str_base_dir=str_base_dir)
    print("Get model paths")
    print("")
    
    print("Start compute structural control nodes FVS, MDS, SC")
    print("")
    
    # Compute structural control nodes for target models
    for i, model_path in enumerate(model_paths):
        model_name = model_path.name
        model_name = model_name.replace('.txt','')
        print(f"{model_name}start")
        input_cnet=str(model_path)
        bn = BN.from_file(input_cnet, type="cnet")
        IG = bn.structural_graph()
        
        #Compute effective graph distinct edge effectiveness
        edge_counts = get_distinct_edge_effectiveness(bn)

        f = str_store_dir + model_name + '/'
        Path(f).mkdir(parents=True, exist_ok=True)
        
        out_path = f + 'edge_effectiveness.csv'
        edge_counts.to_csv(out_path, index=False)
        
        # Compute structural control sets of Interaction graph
        IG = bn.structural_graph()
        df_result=create_ground_truth_control_sets_summary(str_summary_dir=str_summary_dir, sModel=model_name, EG=IG) # Compute structural control sets
        out_file= f + 'st_control_ig.csv'
        df_result.to_csv(out_file, index=False)
    
        # Compute structural control sets of Effective graph, threshold 0.0
        threshold=0.0
        EG = bn.effective_graph(threshold=threshold)
        df_result=create_ground_truth_control_sets_summary(str_summary_dir=str_summary_dir, sModel=model_name, EG=EG)  # Compute structural control sets
        out_file= f + 'st_control_eg_00.csv'
        df_result.to_csv(out_file, index=False)
    
        # Compute structural control sets of Effective graphs, distinct thresholds
        n=1
        for threshold in edge_counts["Edge effectiveness"]:
            if threshold==1.000: break
            
            threshold_dis = threshold + 0.001
            EG = bn.effective_graph(threshold=threshold_dis)
            if EG.number_of_edges() < 1: break
            df_result=create_ground_truth_control_sets_summary(str_summary_dir=str_summary_dir, sModel=model_name, EG=EG)  # Compute structural control sets
        
            out_file = f"{f}st_control_eg_{n:02d}.csv"
            df_result.to_csv(out_file, index=False)
            n=n+1

    print("")
    print("Complete compute structural control nodes FVS, MDS, SC")

def create_ground_truth_control_sets_summary(str_summary_dir, sModel='your model', EG=None, FVS=None, MDS=None, SC=None):

    summary_dir = Path(str_summary_dir)

    model_name_summary = sModel + ' summary'
    csv_file_name = f"{model_name_summary}.csv"
    
    corresponding_csv = summary_dir / csv_file_name
    df = pd.read_csv(corresponding_csv)

    if EG != None:
        FVS = fvs_bruteforce(directed_graph=EG)
        MDS = mds(directed_graph=EG, max_search=20)
        SC = sc(directed_graph=EG)

    id_column = 'node_id' 
    if id_column not in df.columns:
        print('Invalid summary csv')
        return
    else:
        id_series = df[id_column]

        # --- Add FVS information ---
        fvs_cols = []
        if FVS:
            for idx, driver_set in enumerate(FVS, start=1):
                col_name = f"is_FVS_{idx}"
                df[col_name] = id_series.isin(driver_set).astype(int)
                fvs_cols.append(col_name)
            
        # NEW: Aggregated master column for FVS
        # If any of the is_FVS# columns are 1, it results in True, otherwise False
        df['is_FVS'] = df[fvs_cols].any(axis=1) if fvs_cols else False

        # --- Add MDS information ---
        mds_cols = []
        if MDS:
            for idx, driver_set in enumerate(MDS, start=1):
                col_name = f"is_MDS_{idx}"
                df[col_name] = id_series.isin(driver_set).astype(int)
                mds_cols.append(col_name)
            
        # NEW: Aggregated master column for MDS
        df['is_MDS'] = df[mds_cols].any(axis=1) if mds_cols else False

        # --- Add SC information ---
        sc_cols = []
        if SC:
            for idx, driver_set in enumerate(SC, start=1):
                col_name = f"is_SC_{idx}"
                df[col_name] = id_series.isin(driver_set).astype(int)
                sc_cols.append(col_name)
            
        # NEW: Aggregated master column for SC
        df['is_SC'] = df[sc_cols].any(axis=1) if sc_cols else False
         
        # Keep a reference in the dictionary
        #dfs_dict[model_name] = df
    return df