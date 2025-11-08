celltypes=(
  "DC_c3-LAMP3"
  "Macro_c3-EREG"
  "Macro_c4-DNAJb1"
  "Neu_c4-RSAD2"
  "Neu_c6-FGF23"
)

for celltype in "${celltypes[@]}"; do
for thres in 25 30 35; do
# for thres in 0.92; do
  python /Users/liuxinyuan/Desktop/new_projects/cell_retrieval/plot_codes/novel_cell_type/Panel_quantitative_benchmark/COVID/stats_precision_score_ratio.py \
  --cell_type "$celltype" \
  --threshold "$thres"
done;
done