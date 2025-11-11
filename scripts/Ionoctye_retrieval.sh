for j in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity"; do
    for tissue in "lung" "bronchus"; do
        for query_index in "1_ref_2_query" "2_ref_1_query"; do
            for novel in "novel" "not_novel"; do
            python ../retrieval/main.py --input_adata "../data/Ionocyte/ioncyte_${tissue}.h5ad" \
                --input_embeddings "../data/Ionocyte/embeddings/ioncyte_${tissue}_${j}.npy" \
                --method $j \
                --retrieved_for_each_cell 100 \
                --faiss_search L2 \
                --obs cell_type \
                --query_indexes "../data/Ionocyte/${tissue}_query_index_${novel}_${query_index}.npy" \
                --target_indexes "../data/Ionocyte/${tissue}_reference_index_${novel}_${query_index}.npy" \
                --output_dir "../data/Ionocyte/${query_index}/${tissue}/${novel}/${j}"
            done
        done
    done
done

for j in "cellblast" "scvi" "linearscvi" "pca"; do
    for tissue in "lung" "bronchus"; do
        for query_index in "1_ref_2_query" "2_ref_1_query"; do
            for novel in "novel" "not_novel"; do
            python ../retrieval/main.py --input_adata "../data/Ionocyte/ioncyte_${tissue}.h5ad" \
                --input_embeddings "../data/Ionocyte/embeddings/${j}/${query_index}/ioncyte_${tissue}_${j}.npy" \
                --method $j \
                --retrieved_for_each_cell 100 \
                --faiss_search L2 \
                --obs cell_type \
                --query_indexes "../data/Ionocyte/${tissue}_query_index_${novel}_${query_index}.npy" \
                --target_indexes "../data/Ionocyte/${tissue}_reference_index_${novel}_${query_index}.npy" \
                --output_dir "../data/Ionocyte/${query_index}/${tissue}/${novel}/${j}"
            done
        done
    done
done
