#!/bin/bash

# Example command:
# pip install cellplm
# python embeddings.py --input_adata_path "../../raw_data/Limb-Embryo/st.h5ad" --output_embeddings_path  "../../raw_data/Limb-Embryo/embeddings/st_hindlimb_cellplm.npy"


# Define variables for all arguments
INPUT_ADATA_PATH=$1
OUTPUT_EMBEDDINGS_PATH=$2
# Run the Python script with the variable inputs
python embeddings.py \
    --input_adata_path "$INPUT_ADATA_PATH" \
    --output_embeddings_path "$OUTPUT_EMBEDDINGS_PATH" \