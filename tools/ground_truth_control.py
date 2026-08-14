import pandas as pd
import numpy as np
from cana.boolean_network import BooleanNetwork as BN
from pathlib import Path

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
    
def compute_ground_truth_control_sets(str_base_dir, str_store_dir):
    """Compute ground truth control sets (as perturbation attractor driver nodes) of interaction graphs for target models.
    Args:
        str_base_dir  (str): Path to the directory where target Boolean network model text files are stored.
        str_store_dir (str): Path to the directory where summary results will be stored.
    """

    # Get target model paths
    model_paths=get_target_bn_path(str_base_dir=str_base_dir)
    print("Get model paths")
    print("")
    
    print("Start compute ground truth driver node sets")
    print("")
    
    # Compute structural control nodes for target models
    for i, model_path in enumerate(model_paths):
        model_name = model_path.name
        model_name = model_name.replace('.txt','')
        print(f"{model_name}start")
        input_cnet=str(model_path)
        bn = BN.from_file(input_cnet, type="cnet")

        max_dvs=6
        df_result=create_ground_truth_control_sets_summary(bn, max_dvs=max_dvs)
        out_file= str_store_dir + model_name + " summary.csv"
        df_result.to_csv(out_file, index=False)

def create_ground_truth_control_sets_summary(bn, sModel='your model', max_dvs=None):
    """
    Create a DataFrame describing input nodes and perturbation
    driver-node sets.

    Args:
        bn: CANA BooleanNetwork object.
        sModel (str): Model name.
        max_dvs: Maximum number of driver-node sets.

    Returns:
        pandas.DataFrame: Node-level driver/control-set information.
    """

    # Get input nodes
    input_nodes = get_input_nodes(bn)

    # Get perturbation driver-node sets
    driver_nodes = bn.attractor_driver_nodes(max_dvs=max_dvs)

    # Handle empty result
    if driver_nodes is None:
        driver_nodes = []

    # Get node IDs from the Boolean network
    node_ids = [node.id for node in bn.nodes]

    # Create basic DataFrame
    df = pd.DataFrame({
        'node_id': node_ids,
        'node_name': [
            bn.nodes[node_id].name for node_id in node_ids
        ]
    })

    # Mark input nodes
    df['is_input_node'] = (
        df['node_id'].isin(input_nodes)
    )

    # Combine all perturbation driver nodes
    all_driver_nodes = {
        node_id
        for driver_set in driver_nodes
        for node_id in driver_set
    }

    # Mark perturbation driver nodes
    df['is_perturbation_driver'] = (
        df['node_id'].isin(all_driver_nodes)
    )

    # Add one column for each driver set
    for idx, driver_set in enumerate(driver_nodes, start=1):
        col_name = f'is_driver_set_{idx}'

        df[col_name] = (
            df['node_id'].isin(driver_set).astype(int)
        )

    return df
    
def get_input_nodes(bn):

    SG = bn.structural_graph()

    #print("Number of nodes")
    #print(SG.number_of_nodes())
    #print("Number of edges")
    #print(SG.number_of_edges())    

    input_nodes = []
    for n in SG.nodes():
        preds = set(SG.predecessors(n)) - {n}  # ignore self-loop
        if len(preds) == 0:
            input_nodes.append(n)
            
    print("Input nodes")
    print(len(input_nodes))

    return input_nodes

