# Network Control Analysis Pipeline

A Python toolkit for computing ground truth driver node sets, evaluating structural control node sets, and visualizing comparative metrics (e.g., Jaccard Similarity) across large network models.

---

## 📁 Repository Structure

```text
.
├── datasets/                 # Input dataset directory
│   └── targets_test/         # Target network model data
├── tools/                    # Core control analysis modules
│   ├── ground_truth_control.py
│   ├── structural_control.py
│   └── control_sets_comparison.py
├── summary_csv_test/         # Generated ground truth summary files
├── structural_driver_summary_by_test/  # Computed structural control outputs
├── sample_notebook.ipynb     # Demonstration Jupyter Notebook
└── README.md

```

## ⚡ Quick Start
1. Compute Ground Truth Driver Node Sets
Note: For large models, computation can be time-consuming. It is recommended to compute driver node sets individually for each model.

```python
from tools.ground_truth_control import compute_ground_truth_control_sets

str_base_dir = "../datasets/targets_test/"
str_store_dir = "./summary_csv_test/"

compute_ground_truth_control_sets(
    str_base_dir=str_base_dir, 
    str_store_dir=str_store_dir
)
```
2. Compute Structural Control Node Sets
Calculate structural control driver nodes using the pre-computed summary metadata:

```python
from tools.structural_control import compute_structural_control_nodes

str_base_dir = "../datasets/targets_test/"
str_summary_dir = "./summary_csv_test/"
str_store_dir = "./structural_driver_summary_by_test/"

compute_structural_control_nodes(
    str_base_dir=str_base_dir,
    str_summary_dir=str_summary_dir,
    str_store_dir=str_store_dir
)
```
3. Visualize Ground Truth vs. Structural Control Comparisons
Generate comparative grid plots (e.g., Jaccard index) across models:

```python
from tools.control_sets_comparison import visualize_structural_control_nodes_vs_ground_truth

str_reference_path = "./structural_driver_summary_by_test/"
measure_type = "Jaccard"
nrows = 3
ncols = 4

visualize_structural_control_nodes_vs_ground_truth(
    str_reference_path=str_reference_path,
    measure_type=measure_type,
    nrows=nrows,
    ncols=ncols
)
```
## 🚀 Workflow Overview
```text
[ Input Datasets ] ──► compute_ground_truth_control_sets() ──► Ground Truth Summaries
                                                                     │
[ Input Datasets ] ◄── compute_structural_control_nodes() ◄──────────┘
        │
        ▼
Structural Control Summaries ──► visualize_structural_control_nodes_vs_ground_truth() ──► Plots/Evaluation
```

## 💡 Usage Notes & Tips
Large Models: Run compute_ground_truth_control_sets per model in dedicated sub-jobs if working in an HPC / cluster environment to prevent memory bottlenecks.

Plots Grid Configuration: Adjust nrows and ncols in visualize_structural_control_nodes_vs_ground_truth depending on the total number of network models you wish to display on a single canvas.

***

<ElicitationsGroup message="If you'd like to tailor this further, I can help you add:">

<Elicitation label="Add Installation & Requirements section" query="Add an Installation and Requirements section to the README.md, assuming standard packages like numpy, pandas, matplotlib, and networkx."/>

<Elicitation label="Add license, contribution guidelines, or badges" query="Update the README.md to include standard GitHub badges, an MIT license section, and contribution guidelines."/>
