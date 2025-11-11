for method in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity" "pca" "scvi" "linearscvi" "cellblast"; do
python ../retrieval/main.py --input_adata "./raw_data/Limb-Embryo/st_hindlimb.h5ad" \
    --paired_adata "./raw_data/Limb-Embryo/sc_hindlimb.h5ad" \
    --input_embeddings "./raw_data/Limb-Embryo/embeddings/st_hindlimb_${method}.npy" \
    --pair_embeddings "./raw_data/Limb-Embryo/embeddings/sc_hindlimb_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data \
    --obs donor_id,cell_type \
    --output_dir "./raw_data/Limb-Embryo/st_sc/${method}"\

done
