#!/bin/bash

# Example command:
# python train_scvi_paired.py --input_adata ../../raw_data/Arterial-RCA/sc.h5ad --output_data ../../raw_data/Arterial-RCA/embeddings/right_coronary_artery_sc_scvi.npy --pair_adata ../../raw_data/Arterial-RCA/st.h5ad --output_pair_data ../../raw_data/Arterial-RCA/embeddings/right_coronary_artery_st_scvi.npy

# Define variables for all arguments
INPUT_ADATA=$1
OUTPUT_DATA=$2
OUTPUT_PAIR_DATA=$3
PAIR_ADATA=$4
# Run the Python script with the variable inputs
python train_scvi_paired.py \
    --input_adata "$INPUT_ADATA" \
    --output_data "$OUTPUT_DATA" \
    --output_pair_data "$OUTPUT_PAIR_DATA" \
    --pair_adata "$PAIR_ADATA"