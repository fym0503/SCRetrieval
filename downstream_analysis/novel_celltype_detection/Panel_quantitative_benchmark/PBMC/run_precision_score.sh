celltypes=(
  "B cell"
  "CD4+ T cell"
  "CD14+ monocyte"
  "CD16+ monocyte"
  "Cytotoxic T cell"
  "Dendritic cell"
  "Megakaryocyte"
  "Natural killer cell"
  "Plasmacytoid dendritic cell"
)

for celltype in "${celltypes[@]}"; do
for thres in 5 10 15 20 25 30; do
  python /Users/liuxinyuan/Desktop/new_projects/cell_retrieval/plot_codes/novel_cell_type/Panel_quantitative_benchmark/PBMC/stats_precision_score_ratio.py \
  --cell_type "$celltype" \
  --threshold "$thres"
done;
done