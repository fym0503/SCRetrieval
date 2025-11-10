celltypes=(
  "acinar"
  "activated_stellate"
  "alpha"
  "beta"
  "delta"
  "ductal"
  "endothelial"
  "epsilon"
  "gamma"
  "macrophage"
  "mast"
  "quiescent_stellate"
  "schwann"
)

for celltype in "${celltypes[@]}"; do
for thres in 5 10 15 20 25 30; do
# for thres in 0.92; do
  python stats_precision_score_ratio.py \
  --cell_type "$celltype" \
  --threshold "$thres"
done;
done