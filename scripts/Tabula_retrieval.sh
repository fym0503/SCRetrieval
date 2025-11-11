for split in {1..5}; do
for method in "cellblast"; do
for tissue in "large_intestine" "adipose_tissue" "subcutaneous_adipose_tissue" "spleen" "kidney" "thymus" "lung" "liver" "skin_of_body" "trachea"; do
python ../retrieval/main.py --input_adata ../data/Tabula/${tissue}.h5ad \
    --input_embeddings ../data/Tabula/embeddings/${method}_mouse_human_${split}_float/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_train.npy \
    --output_dir ../data/Tabula/results_${method}_${split}/mouse_human_${tissue}/
python ../retrieval/main.py --input_adata ../data/Tabula/${tissue}.h5ad \
    --input_embeddings ../data/Tabula/embeddings/${method}_human_mouse_${split}_float/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_train.npy \
    --output_dir ../data/Tabula/results_${method}_${split}/human_mouse_${tissue}/
done
done
done

for split in {1..5}; do
for method in "scvi" "linear_scvi" "pca"; do
for tissue in "large_intestine" "adipose_tissue" "subcutaneous_adipose_tissue" "spleen" "kidney" "thymus" "lung" "liver" "skin_of_body" "trachea"; do
python ../retrieval/main.py --input_adata ../data/Tabula/${tissue}.h5ad \
    --input_embeddings ../data/Tabula/embeddings/${method}_mouse_human_${split}/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_train.npy \
    --output_dir ../data/Tabula/results_${method}_${split}/mouse_human_${tissue}/
python ../retrieval/main.py --input_adata ../data/Tabula/${tissue}.h5ad \
    --input_embeddings ../data/Tabula/embeddings/${method}_human_mouse_${split}/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_train.npy \
    --output_dir ../data/Tabula/results_${method}_${split}/human_mouse_${tissue}/
done
done
done


for split in {1..5}; do
for method in "scgpt_float" "scimilarity_float" "uce_float" "geneformer_result_float" "scmulan_float" "cellplm_float" "scfoundation_float"; do
for tissue in "large_intestine" "adipose_tissue" "subcutaneous_adipose_tissue" "spleen" "kidney" "thymus" "lung" "liver" "skin_of_body" "trachea"; do
python ../retrieval/main.py --input_adata ../data/Tabula/${tissue}.h5ad \
    --input_embeddings ../data/Tabula/embeddings/${method}/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_mouse_human_${split}/mouse_human_${tissue}_train.npy \
    --output_dir ../data/Tabula/results_${method}_${split}/mouse_human_${tissue}/
python ../retrieval/main.py --input_adata ../data/Tabula/${tissue}.h5ad \
    --input_embeddings ../data/Tabula/embeddings/${method}/${tissue}.npy \
    --method ${method} \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs cell_type,assay \
    --query_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_test.npy \
    --target_indexes ./retrieval/cross_species_human_mouse_${split}/human_mouse_${tissue}_train.npy \
    --output_dir ../data/Tabula/results_${method}_${split}/human_mouse_${tissue}/
done
done
done
