


![figure](../../logo/scretrieval.png)

# Downstream analysis: Marker genes analysis

This repository provides downstream evaluation of marker genes along two dimensions:
- Consistency: stability of detected marker genes across methods.
- Detection accuracy: how well detected markers recover a reference/ground-truth set.

We include scripts and notebooks for reproducible metrics and visualization.

---

## Contents

- `Panel_benchmark_marker_consistency/`
  - `run_all.sh` — running script for consistency analysis across methods. It runs five python scripts:
    - `compute_deg_consistency.py` — calculate jaccard similarity score (JSS) between de genes of each pair of methods.
    - `compute_marker_consistency.py` — calculate jaccard similarity score (JSS) between detected marker genes of each pair of methods.
    - `read.py` -summarize output results to csv file
    - `plot_heatmap_deg.py` - generate plot of deg consistency output
    - `plot_heatmap_marker.py` - generate plot of marker consistency output

- `Panel_benchmark_marker_detection_accuracy/`
  - `run_all.sh` — running script for detection-accuracy analysis with ground truth. It runs two python scripts:
    - `compute_detection_accuracy.py` - calculate precision of predicted de genes
    - `compute_detection_vote.py` - calculate recall of predicted de genes
  - `DE_genes_plots.ipynb` — notebook for visualization

---

## Required data

Download upstream benchmark outputs and place them under downstream_output/:

- Source (Google Drive): https://drive.google.com/drive/folders/10vaR2C0SKWZztVkGlQPF5MqEEpWwBmSq?usp=drive_link
- Expected folders (examples):
  - `downstream_output/ablation/` - stores required data of ablation study and hyperparameter tunning
  - `downstream_output/DEG_analysis/` - stores required data for DE analysis in `Panel_benchmark_marker_consistency` and `Panel_benchmark_marker_detection_accuracy` folder.
  - `downstream_output/novel_cell_type/` - stores required data for novel cell detection analysis in `novel_celltype_detection` folder.
  - `downstream_output/UMAP/` - stores required data for UMAP visualization

These include DE results, candidate marker lists, and cell-type labels used as inputs.

---

## Workflow

Detection consistency:

```bash
sh Panel_benchmark_marker_consistency/run_all.sh
```

Inputs: `downstream_output/DEG_analysis/output/predicted_deg`
Outputs: 
- full_result_deg_overlap.csv
- full_result_marker_overlap.csv
- full_result_deg_overlap.pdf
- full_result_marker_overlap.pdf

```bash
sh Panel_benchmark_marker_detection_accuracy/run_all.sh
```
Inputs: `downstream_output/DEG_analysis/output/predicted_deg`
Outputs: 
- full_result.csv
