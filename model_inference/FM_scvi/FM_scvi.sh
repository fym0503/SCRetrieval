#!/bin/bash

# Example command:

# Define variables for all arguments
INPUT_ADATA=$1 # .adata file
OUTPUT_PATH=$2
FM_EMB=$3 # emb.npy file

# Run the Python script with the variable inputs
python run_FM_scVI.py \
    --input-rna "$INPUT_ADATA" \
    --rna-pre "$FM_EMB" \
    --output-path "$OUTPUT_PATH" \

