#!/bin/bash

# Example command:
# python ./embeddings.py --data_path "../../raw_data/Limb-Embryo/st.h5ad" --output_dir "../../raw_data/Limb-Embryo/embeddings/st_hindlimb_scmulan.npy"

# Define variables for all arguments
DATA_PATH=$1
OUTPUT_DIR=$2
# Run the Python script with the variable inputs
python embeddings.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
