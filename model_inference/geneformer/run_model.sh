#!/bin/bash

# Example command:
# python ./run_geneformer.py --input_adata "../../raw_data/Limb-Embryo/st.h5ad" --output_dir "../../raw_data/Limb-Embryo/embeddings/st_hindlimb_geneformer.npy" --model_save_dir "./Geneformer/geneformer-12L-30M"

# Define variables for all arguments
INPUT_ADATA=$1
OUTPUT_DIR=$2
MODEL_SAVE_DIR=$3
# Run the Python script with the variable inputs
python run_geneformer.py \
    --input_adata "$INPUT_ADATA" \
    --output_dir "$OUTPUT_DIR" \
    --model_save_dir "$MODEL_SAVE_DIR" \