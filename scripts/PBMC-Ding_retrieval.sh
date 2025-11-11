#pbmc
for j in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity"; do
    for ((i=1;i<=9;i+=1));do
        python ../retrieval/main.py --input_adata "../data/PBMC-Ding/PBMC-Ding.h5ad" \
            --input_embeddings "../data/PBMC-Ding/embeddings/embeds_${j}.npy" \
            --method $j \
            --retrieved_for_each_cell 100 \
            --faiss_search L2 \
            --obs CellType,Method \
            --query_indexes "../data/PBMC-Ding/split_${i}_query_index_method.npy" \
            --output_dir "../data/PBMC-Ding/${j}_${i}_method"
    done
done

for j in "pca" "scvi" "linearscvi" "cellblast"; do
    for ((i=1;i<=9;i+=1));do
        python ../retrieval/main.py --input_adata "../data/PBMC-Ding/PBMC-Ding.h5ad" \
            --input_embeddings "../data/PBMC-Ding/embeddings/${j}/method_${i}.npy" \
            --method $j \
            --retrieved_for_each_cell 100 \
            --faiss_search L2 \
            --obs CellType,Method \
            --query_indexes "../data/PBMC-Ding/split_${i}_query_index_method.npy" \
            --output_dir "../data/PBMC-Ding/${j}_${i}_method"
    done
done
