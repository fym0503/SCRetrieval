#!/bin/bash
# =============================================================================
# Cell Retrieval Pipeline - Example Configurations
# =============================================================================

# -----------------------------------------------------------------------------
# Case 1: Single dataset — Use query indexes only (rest = reference)
# -----------------------------------------------------------------------------
python ../retrieval/main.py \
    --input_adata "./data.h5ad" \
    --input_embeddings "./embedding.npy" \
    --method "model" \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs "CellType,Method" \
    --query_indexes "./query_index.npy" \
    --output_dir "./result_single_query"

# -----------------------------------------------------------------------------
# Case 2: Single dataset — Explicit query + reference indexes
# -----------------------------------------------------------------------------
python ../retrieval/main.py \
    --input_adata "./data.h5ad" \
    --input_embeddings "./embedding.npy" \
    --method "model" \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --obs "CellType,Method" \
    --query_indexes "./query_index.npy" \
    --target_indexes "./reference_index.npy" \
    --output_dir "./result_split_indices"

# -----------------------------------------------------------------------------
# Case 3: Paired datasets — Separate query and reference files
# -----------------------------------------------------------------------------
python ../retrieval/main.py \
    --input_adata "./query_data.h5ad" \
    --paired_adata "./reference_data.h5ad" \
    --input_embeddings "./query_embedding.npy" \
    --pair_embeddings "./reference_embedding.npy" \
    --method "model" \
    --retrieved_for_each_cell 100 \
    --faiss_search L2 \
    --norm no \
    --has_paired_data \
    --obs "donor_id,cell_type" \
    --output_dir "./result_paired_data"
