#!/bin/bash

# Example command:

# Define variables for all arguments
INPUT_ADATA=$1 # .adata file
OUTPUT_PATH=$2
PRETRAIN_MODEL=$3 # emb.npy file

# Run the Python script with the variable inputs
python run_scgpt_ft.py \
    --input_data "$INPUT_ADATA" \
    --pre_model "$PRETRAIN_MODEL" \
    --save_dir "$OUTPUT_PATH" \