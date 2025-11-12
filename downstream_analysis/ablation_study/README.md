
# Analysis of different hyperparameters

To test the influence of different hyperparameters of retrieval process using Faiss (n_centroids and n_probe).

We run experiments on SCimilarity and show the analysis code here.

---

## Contents
- `scimilarity_metrics_abl.csv` - Summarized results of ablation experiments
- `ablation_study.py` - Visualization of ablation experiments results
- `umap_and_abl.ipynb` - notebook for UMAP visualization and extended plot

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
