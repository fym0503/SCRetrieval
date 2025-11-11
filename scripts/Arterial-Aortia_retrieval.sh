for method in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity" "pca" "scvi" "linearscvi" "cellblast"; do
python ../retrieval/main.py --input_adata "../data/Arterial-Aortia/aorta_st.h5ad" \
    --paired_adata "../data/Arterial-Aortia/aorta_sc.h5ad" \
    --input_embeddings "../data/Arterial-Aortia/embeddings/aorta_st_${method}.npy" \
    --pair_embeddings "../data/Arterial-Aortia/embeddings/aorta_sc_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data \
    --obs donor_id,cell_type \
    --output_dir "../data/Arterial-Aortia/st_sc/${method}"\

done
