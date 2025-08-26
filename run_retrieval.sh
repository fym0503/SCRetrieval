#pbmc
for j in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity"; do
    for ((i=1;i<=9;i+=1));do
        python main.py --input_adata "./raw_data/PBMC-Ding/adata_pbmc_benchmark.h5ad" \
            --input_embeddings "./raw_data/PBMC-Ding/embeddings/embeds_${j}.npy" \
            --method $j \
            --retrieved_for_each_cell 100 \
            --faiss_search L2 \
            --obs CellType,Method \
            --query_indexes "./raw_data/PBMC-Ding/split_${i}_query_index_method.npy" \
            --output_dir "./raw_data/PBMC-Ding/${j}_${i}_method"
    done
done

for j in "pca" "scvi" "linearscvi" "cellblast"; do
    for ((i=1;i<=9;i+=1));do
        python main.py --input_adata "./raw_data/PBMC-Ding/adata_pbmc_benchmark.h5ad" \
            --input_embeddings "./raw_data/PBMC-Ding/embeddings/${j}/method_${i}.npy" \
            --method $j \
            --retrieved_for_each_cell 100 \
            --faiss_search L2 \
            --obs CellType,Method \
            --query_indexes "./raw_data/PBMC-Ding/split_${i}_query_index_method.npy" \
            --output_dir "./raw_data/PBMC-Ding/${j}_${i}_method"
    done
done
#pancreas

for j in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity" "pca" "scvi" "linearscvi" "cellblast"; do
    for ((i=1;i<=4;i+=1));do
        python main.py --input_adata "./raw_data/Pancreas/pancreas.h5ad" \
            --input_embeddings "./raw_data/Pancreas/embeddings/embeds_${j}.npy" \
            --method $j \
            --retrieved_for_each_cell 100 \
            --faiss_search L2 \
            --obs CellType,Method \
            --query_indexes "./raw_data/Pancreas/split_${i}_query_index_method.npy" \
            --output_dir "./raw_data/Pancreas/${j}_${i}_method"
    done
done

for j in "pca" "scvi" "linearscvi" "cellblast"; do
    for ((i=1;i<=4;i+=1));do
        python main.py --input_adata "./raw_data/Pancreas/pancreas.h5ad" \
            --input_embeddings "./raw_data/Pancreas/embeddings/${j}/method_${i}.npy" \
            --method $j \
            --retrieved_for_each_cell 100 \
            --faiss_search L2 \
            --obs CellType,Method \
            --query_indexes "./raw_data/Pancreas/split_${i}_query_index_method.npy" \
            --output_dir "./raw_data/Pancreas/${j}_${i}_method"
    done
done

#tissues
#use uce data
for split in {1..5}; do
# for method in "scgpt_float" "scimilarity_float" "uce_float" "geneformer_result_float" "scmulan_float" "cellplm_float"; do
for method in "cellblast"; do
for tissue in "large_intestine" "adipose_tissue" "subcutaneous_adipose_tissue" "spleen" "kidney" "thymus" "lung" "liver" "skin_of_body" "trachea"; do
python main.py --input_adata ./raw_data/Tabula/${tissue}.h5ad \
    --input_embeddings ./raw_data/Tabula/embeddings/${method}_mouse_human_${split}_float/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_train.npy \
    --output_dir ./raw_data/Tabula/results_${method}_${split}/mouse_human_${tissue}/
python main.py --input_adata ./raw_data/Tabula/${tissue}.h5ad \
    --input_embeddings ./raw_data/Tabula/embeddings/${method}_human_mouse_${split}_float/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_train.npy \
    --output_dir ./raw_data/Tabula/results_${method}_${split}/human_mouse_${tissue}/
done
done
done

for split in {1..5}; do
# for method in "scgpt_float" "scimilarity_float" "uce_float" "geneformer_result_float" "scmulan_float" "cellplm_float"; do
for method in "scvi" "linear_scvi" "pca"; do
for tissue in "large_intestine" "adipose_tissue" "subcutaneous_adipose_tissue" "spleen" "kidney" "thymus" "lung" "liver" "skin_of_body" "trachea"; do
python main.py --input_adata ./raw_data/Tabula/${tissue}.h5ad \
    --input_embeddings ./raw_data/Tabula/embeddings/${method}_mouse_human_${split}/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_train.npy \
    --output_dir ./raw_data/Tabula/results_${method}_${split}/mouse_human_${tissue}/
python main.py --input_adata ./raw_data/Tabula/${tissue}.h5ad \
    --input_embeddings ./raw_data/Tabula/embeddings/${method}_human_mouse_${split}/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_train.npy \
    --output_dir ./raw_data/Tabula/results_${method}_${split}/human_mouse_${tissue}/
done
done
done


for split in {1..5}; do
for method in "scgpt_float" "scimilarity_float" "uce_float" "geneformer_result_float" "scmulan_float" "cellplm_float" "scfoundation_float"; do
for tissue in "large_intestine" "adipose_tissue" "subcutaneous_adipose_tissue" "spleen" "kidney" "thymus" "lung" "liver" "skin_of_body" "trachea"; do
python main.py --input_adata ./raw_data/Tabula/${tissue}.h5ad \
    --input_embeddings ./raw_data/Tabula/embeddings/${method}/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_train.npy \
    --output_dir ./raw_data/Tabula/results_${method}_${split}/mouse_human_${tissue}/
python main.py --input_adata ./raw_data/Tabula/${tissue}.h5ad \
    --input_embeddings ./raw_data/Tabula/embeddings/${method}/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_train.npy \
    --output_dir ./raw_data/Tabula/results_${method}_${split}/human_mouse_${tissue}/
done
done
done

#cross species
for cross in "human_mouse" "mouse_human"; do
IFS="_" read -r species1 species2 <<< "$cross"
for method in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity"; do
# for norm_name in "standard" "sc1p" "no"; do
python main.py --input_adata "./raw_data/Cerebellum-dev/${species1}.h5ad" \
    --paired_adata "./raw_data/Cerebellum-dev/${species2}.h5ad" \
    --input_embeddings "./raw_data/Cerebellum-dev/embeddings/${species1}_cerebellum_${method}.npy" \
    --pair_embeddings "./raw_data/Cerebellum-dev/embeddings/${species2}_cerebellum_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data yes \
    --obs assay,cell_type \
    --output_dir "./raw_data/Cerebellum-dev/${cross}/${method}"\

done
done


for cross in "human_mouse" "mouse_human"; do
IFS="_" read -r species1 species2 <<< "$cross"
for method in "pca" "scvi" "linearscvi" "cellblast"; do
# for norm_name in "standard" "sc1p" "no"; do
python main.py --input_adata "./raw_data/Cerebellum-dev/${species1}.h5ad" \
    --paired_adata "./raw_data/Cerebellum-dev/${species2}.h5ad" \
    --input_embeddings "./raw_data/Cerebellum-dev/embeddings/${cross}_${method}/${species1}_cerebellum_${method}.npy" \
    --pair_embeddings "./raw_data/Cerebellum-dev/embeddings/${cross}_${method}/${species2}_cerebellum_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data yes \
    --obs assay,cell_type \
    --output_dir "./raw_data/Cerebellum-dev/${cross}/${method}"\

done
done




for cross in "human_mouse" "mouse_human"; do
IFS="_" read -r species1 species2 <<< "$cross"
for method in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity"; do
# for norm_name in "standard" "sc1p" "no"; do
python main.py --input_adata "./raw_data/Muscle-aging/${species1}.h5ad" \
    --paired_adata "./raw_data/Muscle-aging/${species2}.h5ad" \
    --input_embeddings "./raw_data/Muscle-aging/embeddings/${species1}_clean_${method}.npy" \
    --pair_embeddings "./raw_data/Muscle-aging/embeddings/${species2}_clean_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data yes \
    --obs assay,cell_type \
    --output_dir "./raw_data/Muscle-aging/${cross}/${method}"\

done
done


for cross in "human_mouse" "mouse_human"; do
IFS="_" read -r species1 species2 <<< "$cross"
for method in "pca" "scvi" "linearscvi" "cellblast"; do
# for norm_name in "standard" "sc1p" "no"; do
python main.py --input_adata "./raw_data/Muscle-aging/${species1}.h5ad" \
    --paired_adata "./raw_data/Muscle-aging/${species2}.h5ad" \
    --input_embeddings "./raw_data/Muscle-aging/embeddings/${cross}_${method}/${species1}_clean_${method}.npy" \
    --pair_embeddings "./raw_data/Muscle-aging/embeddings/${cross}_${method}/${species2}_clean_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data yes \
    --obs assay,cell_type \
    --output_dir "./raw_data/Muscle-aging/${cross}/${method}"\

done
done


#cross technology
for method in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity" "pca" "scvi" "linearscvi" "cellblast"; do
# for norm_name in "standard" "sc1p" "no"; do
python main.py --input_adata "./raw_data/Arterial-RCA/st.h5ad" \
    --paired_adata "./raw_data/Arterial-RCA/sc.h5ad" \
    --input_embeddings "./raw_data/Arterial-RCA/embeddings/right_coronary_artery_st_${method}.npy" \
    --pair_embeddings "./raw_data/Arterial-RCA/embeddings/right_coronary_artery_sc_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data yes \
    --obs donor_id,cell_type \
    --output_dir "./raw_data/Arterial-RCA/st_sc/${method}"\

done

for method in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity" "pca" "scvi" "linearscvi" "cellblast"; do
# for norm_name in "standard" "sc1p" "no"; do
python main.py --input_adata "./raw_data/Arterial-Aortia/st.h5ad" \
    --paired_adata "./raw_data/Arterial-Aortia/sc.h5ad" \
    --input_embeddings "./raw_data/Arterial-Aortia/embeddings/aorta_st_${method}.npy" \
    --pair_embeddings "./raw_data/Arterial-Aortia/embeddings/aorta_sc_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data yes \
    --obs donor_id,cell_type \
    --output_dir "./raw_data/Arterial-Aortia/st_sc/${method}"\

done

for method in "cellplm" "geneformer" "scgpt" "scimilarity" "scmulan" "uce" "scimilarity" "pca" "scvi" "linearscvi" "cellblast"; do
# for norm_name in "standard" "sc1p" "no"; do
python main.py --input_adata "./raw_data/Limb-Embryo/st.h5ad" \
    --paired_adata "./raw_data/Limb-Embryo/sc.h5ad" \
    --input_embeddings "./raw_data/Limb-Embryo/embeddings/st_hindlimb_${method}.npy" \
    --pair_embeddings "./raw_data/Limb-Embryo/embeddings/sc_hindlimb_${method}.npy" \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data yes \
    --obs donor_id,cell_type \
    --output_dir "./raw_data/Limb-Embryo/st_sc/${method}"\

done

