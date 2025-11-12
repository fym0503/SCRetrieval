
# Downstream analysis: Novel cell type detection

This repository provides downstream evaluation of novel cell type detection, including:
- Quantitative benchmark using accuracy, recall, precision, f1 score, AUROC score.
- Statistical analysis.

We include scripts and notebooks for reproducible metrics and visualization.

---

## Contents

- `Panel_quantitative_benchmark/`
  - `COVID`
  - `Ionocyte`
  - `Pancreas`
  - `PBMC`
    - `run_precision_score.sh` — running script for quantitative evaluation with accuracy, precison, recall, and f1 score. It runs `stats_precision_score_ratio.py`.
    - `submit.py` — running script for quantitative evaluation with with AUROC score. It runs `stats_auroc_score.py`.
    - `stats_scores_different_range.py` - summarize auroc score results

- `Panel_statistical_testing/`
  - `performance_test_heatmap_pancreas.py`- do wilcoxon test of metrics output and generate heatmap plot for pancreas dataset
  - `performance_test_heatmap_pbmc.py` - do wilcoxon test of metrics output and generate heatmap plot for PBMC dataset

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

#### Quantitative benchmark (take COVID as an example):

```bash
bash Panel_quantitative_benchmark/COVID/run_precision_score.sh
```

Inputs: 

`downstream_output/novel_cell_type/disease_novel/`

Outputs: 

`csv files of each metrics`


#### Statistical test

```bash
python Panel_statistical_testing/performance_test_heatmap_pancreas.py
python Panel_statistical_testing/performance_test_heatmap_pbmc.py
```

Inputs: 

`output csv files from Quantitative benchmark`

Outputs: 

`heatmaps of pancreas and PBMC datasets`

#### Visualization
Run `novelcell_plots.ipynb` notebook.
