for method in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity" "pca" "scvi" "linearscvi" "cellblast"; do
python ../retrieval/main.py --input_adata "../data/COVID/disease_df_not_novel.h5ad" \
    --paired_adata "../data/COVID/healthy.h5ad" \
    --input_embeddings "../data/COVID/embeddings/disease_df_not_novel_${method}.npy" \
    --pair_embeddings "../data/COVID/embeddings/healthy_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data \
    --obs celltype \
    --output_dir "../data/COVID/disease_not_novel/${method}" \

done

for method in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity" "pca" "scvi" "linearscvi" "cellblast"; do
python ../retrieval/main.py --input_adata "../data/COVID/disease_df_novel.h5ad" \
    --paired_adata "../data/COVID/healthy.h5ad" \
    --input_embeddings "../data/COVID/embeddings/disease_df_novel_${method}.npy" \
    --pair_embeddings "../data/COVID/embeddings/healthy_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data \
    --obs celltype \
    --output_dir "../data/COVID/disease_novel/${method}" \
    
done
