#!/bin/bash

# Example command:
# python embeddings_paired.py --input_adata "../../raw_data/Limb-Embryo/sc.h5ad" --output_adata "../../raw_data/Limb-Embryo/embeddings/sc_hindlimb_cellblast.npy" --output_pair_adata "../../raw_data/Limb-Embryo/st.npy" --paired_adata "./../raw_data/Limb-Embryo/embeddings/st_hindlimb_all_slides.h5ad"

# Define variables for all arguments
INPUT_ADATA=$1
OUTPUT_ADATA=$2
OUTPUT_PAIR_ADATA=$3
PAIRED_ADATA=$4
# Run the Python script with the variable inputs
python embeddings_paired.py \
    --input_adata "$INPUT_ADATA" \
    --output_adata "$OUTPUT_ADATA" \
    --output_pair_adata "$OUTPUT_PAIR_ADATA" \
    --paired_adata "$PAIRED_ADATA"