#!/bin/bash

# Example command:
# python ./run_uce.py --input_adata "../../raw_data/Limb-Embryo/st.h5ad" --output_dir "../../raw_data/Limb-Embryo/embeddings/st_hindlimb_uce.npy" --model_loc --species "human"

# Define variables for all arguments
INPUT_ADATA=$1
OUTPUT_DIR=$2
MODEL_LOC=$3
SPECIES=$4
# Run the Python script with the variable inputs
python run_uce.py \
    --input_adata "$INPUT_ADATA" \
    --output_dir "$OUTPUT_DIR" \
    --model_loc "$MODEL_LOC" \
    --species "$SPECIES" \