#from cana.boolean_network import BooleanNetwork
import pandas as pd
import itertools
from itertools import combinations
from collections import Counter
import networkx as nx
from cana.boolean_network import BooleanNetwork as BN
from pathlib import Path

def actionability_per_node(str_targets_dir, str_summary_dir, str_store_dir, process_type):
    """`Summarize edge and/or node level actionability by Boolean networks defined in a directory.
        Args:
            str_targets_dir (str): Path to a directory in which cana accesible Boolean network models (.txt) are stored.  
            str_summary_dir (str): Path to a directory in which summary information of Boolean network models are stored.
            str_store_dir (str):   Path to a directory in which redults are stored.
            process_type (str):    Specify type of actionability computations (Node, Edge, or Both)
    """
    # 0. Validate process_type input (Normalize to title case)
    process_type = process_type.capitalize()  # Converts 'node' or 'NODE' to 'Node'
    valid_types = {"Node", "Edge", "Both"}

    if process_type not in valid_types:
        print(f"Error: Invalid process_type '{process_type}'.")
        print(f"Allowed options are: {', '.join(sorted(valid_types))}")
        return

    # Dictionary to hold the dataframes for later use: { "model_name": dataframe }
    dfs_dict = {}

    targets_dir = Path(str_targets_dir)
    summary_dir = Path(str_summary_dir)
    store_dir = Path(str_store_dir+process_type+'/')
    
    # 1. Check if target directory exists
    if not targets_dir.exists():
        print(f"Error: The directory {targets_dir} does not exist.")
        return  # Early exit if directory is missing
    
    txt_files = list(targets_dir.glob("*.txt"))   
    print(f"Found {len(txt_files)} text files in {targets_dir}\n")

    # 2. Check if summary directory exists
    if not summary_dir.exists():
        print(f"Error: The directory {summary_dir} does not exist.")
        return  # Early exit if directory is missing
    
    # 3. Check if both target BN txt and BN summary correspondingly exist
    valid_file_pairs = []
    missing_summaries = []

    for txt_file in txt_files:
        base_name = txt_file.stem        
        # Construct the expected summary filename
        expected_csv_name = f"{base_name} summary.csv"
        expected_csv_path = summary_dir / expected_csv_name        
        # Check if the summary file actually exists
        if expected_csv_path.exists():
            valid_file_pairs.append((txt_file, expected_csv_path))
        else:
            missing_summaries.append(txt_file.name)

    # Report results
    #print(f"Successfully matched {len(valid_file_pairs)} out of {len(txt_files)} file pairs.")
    
    if missing_summaries:
        print(f"Warning: Missing summary files for the following target networks:")
        for missing in missing_summaries:
            print(f"  - {missing}")
        print() # Add a newline for formatting
        return
       
    # 4. Check if store directory exists, create it if it doesn't
    if not store_dir.exists():
        print(f"Directory {store_dir} does not exist. Creating it now...")
        store_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Store directory {store_dir} already exists.")

    # 5. Compute actionability for each model. 
    for txt_file in txt_files:
        bn = BN.from_file(txt_file, type='cnet', keep_constants=False)
        model_name = txt_file.stem
        model_name_summary = model_name + ' summary'
        
        # 3. Construct the matching CSV file path
        csv_file_name = f"{model_name_summary}.csv"
        corresponding_csv = summary_dir / csv_file_name

        #try:
        # Read into pandas DataFrame
        df = pd.read_csv(corresponding_csv)
        dfs_dict[model_name] = df
                
        # Filter the DataFrame and extract node_ids to lists
        # Since the columns are boolean, df['column_name'] == True filters for True values
        perturbation_driver = df[df['is_perturbation_driver'] == True]['node_id'].tolist()
        input_node = df[df['is_input_node'] == True]['node_id'].tolist()

        # --- ACTIONABILITY COMPUTATION LOGIC ---
        node_summary, node_combined = None, None
        edge_summary, edge_combined = None, None
        # Run Node calculation if requested
        if process_type in ('Node', 'Both'):
            print(f"[{model_name}] Computing Node Actionability...")
            node_summary, node_combined = node_actionabulity_per_node(
                bn, perturbation_driver, input_node, False
            )

        # Run Edge calculation if requested
        if process_type in ('Edge', 'Both'):
            print(f"[{model_name}] Computing Edge Actionability...")
            edge_summary, edge_combined = edge_actionabulity_per_node(
                bn, perturbation_driver, input_node, False
            )
        # --- SAVING LOGIC ---
        # Now you decide what to do with the results based on process_type
        if process_type == 'Node':
            summary_out = store_dir / f"{model_name}_node_actionability_summary.csv"
            combined_out = store_dir / f"{model_name}_node_actionability_combined.csv"
            node_summary.to_csv(summary_out, index=False)
            node_combined.to_csv(combined_out, index=False)
            print(f"[{model_name}] Saved Node results to {store_dir}")

        elif process_type == 'Edge':
            summary_out = store_dir / f"{model_name}_edge_actionability_summary.csv"
            combined_out = store_dir / f"{model_name}_edge_actionability_combined.csv"
            edge_summary.to_csv(summary_out, index=False)
            edge_combined.to_csv(combined_out, index=False)
            print(f"[{model_name}] Saved Edge results to {store_dir}")

        elif process_type == 'Both':
            # Identify the shared identifier columns
            shared_cols = [
                'node_id', 'node_name', 'driver_node', 
                'node_role', 'condition_state'
            ]

            # 1. Merge the summaries on the shared columns
            # suffixes differentiate any other columns that happen to share a name
            combined_summary = pd.merge(
                node_summary, 
                edge_summary, 
                on=shared_cols, 
                suffixes=('_node', '_edge')
            )

            shared_cols = [
                'node_id', 'node_name', 'driver_node', 
                'node_role'
            ]
            # 2. For the 'combined' DataFrames, you likely want to do the same 
            # (Assuming they share the same row structure and ID columns)
            combined_data = pd.merge(
                node_combined, 
                edge_combined, 
                on=shared_cols, 
                suffixes=('_node', '_edge')
            )

            # 3. Construct paths for the unified files
            summary_out = store_dir / f"{model_name}_actionability_summary.csv"
            combined_out = store_dir / f"{model_name}_actionability_combined.csv"
                
            # 4. Export to single CSVs
            combined_summary.to_csv(summary_out, index=False)
            combined_data.to_csv(combined_out, index=False)

def edge_actionabulity_per_node(bn, driver_nodes_list=None, input_nodes=[], is_nested=True):
    """
    Calculates edge actionability metrics per individual node in a model, bn.
    Args:
        bn (cana BooleanNetwork object): a target Boolean network model       
        driver_nodes_list (list or list of list): a list of node ids of driver nodes.
        input_nodes (list): a list of node ids of input nodes.
        is_nested (bool): True indicates driver_nodes_list is list of list, False indicates driver_nodes_list is list.
    Returns:
        df_summary (pandas dataframe): summary table of edge level actionability of each node as either conditioned as ON and OFF.
        df_combined (pandas dataframe): summary table of edge level actionability of each node as max, min, and average of two actionability measures in df_summary.
    """
    # 1. Setup metadata
    if driver_nodes_list is None:
        flattened_drivers = set()
    elif is_nested:
        flattened_drivers = {node_id for sublist in driver_nodes_list for node_id in sublist}
    else:
        flattened_drivers = set(driver_nodes_list)
    #flattened_drivers = {node_id for sublist in driver_nodes_list for node_id in sublist}
    input_node_set = set(input_nodes) if input_nodes else set()
    
    nodes = bn.nodes
    IG = bn.structural_graph()
    IG_edges_count = IG.number_of_edges()
    summary_data = []

    out_details=True

    for i in range(len(nodes)):
        target_node_id = nodes[i].id
        node_name = nodes[i].name
        
        node_type = 'driver' if target_node_id in flattened_drivers else 'non-driver'
        node_role = 'input node' if target_node_id in input_node_set else 'inner node'

        for cond_state in range(2):
            conditioned_nodes = {target_node_id: cond_state}
            # Threshold=0 keeps edges in the graph but sets weight to 0.0
            
            actionability, e_on_count, e_off_count, e_redundant_count = compute_edge_actionabulity(bn, conditioned_nodes, out_details)
    
            summary_data.append({
                'node_id': target_node_id,
                'node_name': node_name,
                'driver_node': node_type,
                'node_role': node_role,
                'IG_edges': IG_edges_count,
                'condition_state': cond_state,
                'E_redundant': e_redundant_count,
                'E_ON': e_on_count,
                'E_OFF': e_off_count,
                'edge_actionability': round(actionability, 4)
            })

    # --- Data Aggregation ---
    df_summary = pd.DataFrame(summary_data)

    grouped = df_summary.groupby(['node_id', 'node_name', 'driver_node', 'node_role', 'IG_edges'])
    
    df_combined = grouped.agg({
        'edge_actionability': ['mean', 'min', 'max'],
        'E_redundant': ['mean', 'max', 'min'],
        'E_ON': ['mean', 'max', 'min'],
        'E_OFF': ['mean', 'max', 'min']
    })

    df_combined.columns = [
        'avg_edge_actionability', 'min_edge_actionability', 'max_edge_actionability',
        'avg_E_redundant', 'max_E_redundant', 'min_E_redundant',
        'avg_E_ON', 'max_E_ON', 'min_E_ON',
        'avg_E_OFF', 'max_E_OFF', 'min_E_OFF'
    ]
    
    df_combined = df_combined.reset_index().round(4)

    return df_summary, df_combined

def compute_edge_actionabulity(bn, conditioned_nodes, out_details=False):
    """`Compute edge level actionability.
        Args:
            bn : cana BooleanNetwork object: a target Boolean network model       
            conditioned_nodes (dict) : a dictionary mapping node ids to their conditioned states.
                dict of form { nodeid : nodestate }
            out_details (bool, optional) : Return additional edge information. Defaults to ``False``.

        Returns:
            (float) : edge level actionability.
    """
    # 1. Compute the conditional effective graph
    IG = bn.structural_graph()
    EG_CN = bn.conditional_effective_graph(conditioned_nodes=conditioned_nodes, threshold=0.0)
    
    # 2. Calculate global counts for this specific condition
    IG_edges_count = IG.number_of_edges()
    EG_CN_edges_count = EG_CN.number_of_edges()

    # 3. Difference indicates edges that were removed/simplified
    total_fully_redundant = IG_edges_count - EG_CN_edges_count
    
    # 4. Calculate E_ON and E_OFF (Edge-level resolution)
    # We find nodes resolved to a state and count their OUTGOING edges
    e_on_count = 0
    e_off_count = 0
        
    for nid, d in EG_CN.nodes(data=True):
        c_state = d.get('conditioned_state', None)
            
        if c_state == 1:
            # Count all outgoing edges from this ON-resolved node
            e_on_count += EG_CN.out_degree(nid)
        elif c_state == 0:
            # Count all outgoing edges from this OFF-resolved node
            e_off_count += EG_CN.out_degree(nid)

    # 5. Calculate Reduced Viability (r)
    # r = (|E| - (|E_on| + |E_off| + |E_redundant|)) / |E|
    viable_edges = IG_edges_count - (e_on_count + e_off_count + total_fully_redundant)
    edge_actionabulity = viable_edges / IG_edges_count if IG_edges_count > 0 else 0

    if out_details==True:
        return edge_actionabulity, e_on_count, e_off_count, total_fully_redundant
    else:
        return edge_actionabulity

def node_actionabulity_per_node(bn, driver_nodes_list, input_nodes=[], is_nested=True):
    """
    Calculates node actionability metrics per individual node in a model, bn.
    Args:
        bn (cana BooleanNetwork object): a target Boolean network model       
        driver_nodes_list (list or list of list): a list of node ids of driver nodes.
        input_nodes (list): a list of node ids of input nodes.
        is_nested (bool): True indicates driver_nodes_list is list of list, False indicates driver_nodes_list is list.
    Returns:
        df_summary (pandas dataframe): summary table of node level actionability of each node as either conditioned as ON and OFF.
        df_combined (pandas dataframe): summary table of node level actionability of each node as max, min, and average of two actionability measures in df_summary.
    """
    # 1. Setup metadata
    if is_nested:
        flattened_drivers = {node_id for sublist in driver_nodes_list for node_id in sublist}
    else:
        flattened_drivers = set(driver_nodes_list)
        
    input_node_set = set(input_nodes) if input_nodes else set()
    
    nodes = bn.nodes
    IG = bn.structural_graph()
    IG_nodes_count = IG.number_of_nodes()
    summary_data = []

    for i in range(len(nodes)):
        target_node_id = nodes[i].id
        node_name = nodes[i].name
        
        node_type = 'driver' if target_node_id in flattened_drivers else 'non-driver'
        node_role = 'input node' if target_node_id in input_node_set else 'inner node'

        for cond_state in range(2):
            conditioned_nodes = {target_node_id: cond_state}
            # Threshold=0 keeps edges in the graph but sets weight to 0.0
            EG_CN = bn.conditional_effective_graph(conditioned_nodes=conditioned_nodes)

            node_actionabulity, count_0, count_1 = compute_node_actionabulity(bn, conditioned_nodes, out_details=True)
    
            summary_data.append({
                'node_id': target_node_id,
                'node_name': node_name,
                'driver_node': node_type,
                'node_role': node_role,
                'IG_nodes': IG_nodes_count,
                'condition_state': cond_state,
                'N_ON': count_1,
                'N_OFF': count_0,
                'node_actionability': round(node_actionabulity, 4)
            })

    # --- Data Aggregation ---
    df_summary = pd.DataFrame(summary_data)

    grouped = df_summary.groupby(['node_id', 'node_name', 'driver_node', 'node_role', 'IG_nodes'])
    
    df_combined = grouped.agg({
        'node_actionability': ['mean', 'min', 'max'],
        'N_ON': ['mean', 'max', 'min'],
        'N_OFF': ['mean', 'max', 'min']
    })

    df_combined.columns = [
        'avg_node_actionability', 'min_node_actionability', 'max_node_actionability',
        'avg_N_ON', 'max_N_ON', 'min_N_ON',
        'avg_N_OFF', 'max_N_OFF', 'min_N_OFF'
    ]
    
    df_combined = df_combined.reset_index().round(4)

    return df_summary, df_combined

def compute_node_actionabulity(bn, conditioned_nodes, out_details=False):
    """`Compute node level actionability.
        Args:
            bn : cana BooleanNetwork object: a target Boolean network model       
            conditioned_nodes (dict): a dictionary mapping node ids to their conditioned states.
                dict of form { nodeid : nodestate }
            out_details (bool, optional): Return additional node information. Defaults to ``False``.

        Returns:
            (float) : node level actionability.
    """
    # Initialize variables
    node_actionabulity=0
    node_fixation_ratio=0
    count_0 = 0
    count_1 = 0
    nodes=bn.nodes
    nNodes=len(nodes)
    nCondition_nodes=len(conditioned_nodes)

    # Check conditioning
    if nNodes==nCondition_nodes:
        # If all nodes are conditioned, return result as all nodes are fixed
        node_fixation_ratio=1.0
        node_actionabulity=0.0
        if out_details == False:
            return node_actionabulity
        else:
            counts = Counter(conditioned_nodes.values())
            count_0=counts[0]
            count_1=counts[1]
            return node_actionabulity, count_0, count_1
        
    # Compute the conditional effective graph
    EG_CN = bn.conditional_effective_graph(conditioned_nodes=conditioned_nodes)

    # Inner loop: Count how many OTHER nodes were fixed
    for nid in EG_CN.nodes():
        if nid in conditioned_nodes:
            continue
    
        c_state = EG_CN.nodes[nid].get('conditioned_state', None)
                
        if c_state == 0:
            count_0 += 1
        elif c_state == 1:
            count_1 += 1
    
    # Compute node actionability
    node_fixation_ratio = (count_0 + count_1) / (nNodes-nCondition_nodes)
    node_actionabulity = 1 - node_fixation_ratio

    if out_details==True:
        return node_actionabulity, count_0, count_1
    else:
        return node_actionabulity

def aggregate_actionability_summary(str_summary_dir):
    """`Aggretate edge and/or node level actionability summary files.
        Args:
            str_summary_dir (str): Path to a directory in which summary information of Boolean network models are stored.
        Returns:
            (pandas dataframe)
    """
    summary_dir = Path(str_summary_dir)
    # Check if target directory exists
    if not summary_dir.exists():
        print(f"Error: The directory {summary_dir} does not exist.")
        return  # Early exit if directory is missing

    #search_pattern = os.path.join(folder_path, "*summary.csv")
    #csv_files = glob.glob(search_pattern)

    csv_files = list(summary_dir.glob("*summary.csv"))   
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return None

    print(f"Found {len(csv_files)} text files in {summary_dir}\n")

    df_list = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        
        # This adds the column to every individual dataframe
        df['source_file'] = str(file_path)
        
        df_list.append(df)

    df_aggregate = pd.concat(df_list, ignore_index=True)
    
    print(f"Aggregated {len(csv_files)} files. Total rows: {len(df_aggregate)}")
    
    # Added 'source_file' to the beginning of the list
    ordered_cols = [
        'source_file', 'node_id', 'node_name', 'driver_node', 'node_role', 
        'condition_state', 'IG_nodes', 'IG_edges', 'E_redundant', 'E_ON', 
        'E_OFF', 'edge_actionability', 'N_OFF', 'N_ON', 'node_actionability'
    ]
    
    # 2. This filter now includes 'source_file' because it exists in df_combined_all
    existing_cols = [c for c in ordered_cols if c in df_aggregate.columns]
    df_aggregate = df_aggregate[existing_cols]
    
    return df_aggregate