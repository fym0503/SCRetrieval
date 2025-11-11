for cross in "human_mouse" "mouse_human"; do
IFS="_" read -r species1 species2 <<< "$cross"
for method in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity"; do
python ../retrieval/main.py --input_adata "./raw_data/Cerebellum-dev/${species1}.h5ad" \
    --paired_adata "./raw_data/Cerebellum-dev/${species2}.h5ad" \
    --input_embeddings "./raw_data/Cerebellum-dev/embeddings/${species1}_cerebellum_${method}.npy" \
    --pair_embeddings "./raw_data/Cerebellum-dev/embeddings/${species2}_cerebellum_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data \
    --obs assay,cell_type \
    --output_dir "./raw_data/Cerebellum-dev/${cross}/${method}"\

done
done


for cross in "human_mouse" "mouse_human"; do
IFS="_" read -r species1 species2 <<< "$cross"
for method in "pca" "scvi" "linearscvi" "cellblast"; do
python ../retrieval/main.py --input_adata "./raw_data/Cerebellum-dev/${species1}.h5ad" \
    --paired_adata "./raw_data/Cerebellum-dev/${species2}.h5ad" \
    --input_embeddings "./raw_data/Cerebellum-dev/embeddings/${cross}_${method}/${species1}_cerebellum_${method}.npy" \
    --pair_embeddings "./raw_data/Cerebellum-dev/embeddings/${cross}_${method}/${species2}_cerebellum_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data \
    --obs assay,cell_type \
    --output_dir "./raw_data/Cerebellum-dev/${cross}/${method}"\

done
done