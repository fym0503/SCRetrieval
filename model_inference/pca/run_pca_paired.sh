

#!/bin/bash

# Example command:
# python run_pca_paired.py --input_adata "../../raw_data/Arterial-RCA/sc.h5ad"\
#                        --paired_adata "../../raw_data/Arterial-RCA/st.h5ad"\
#                      --output_adata "../../raw_data/Arterial-RCA/embeddings/sc_hindlimb_pca.npy" \
#                      --output_pair_adata "../../raw_data/Arterial-RCA/embeddings/st_hindlimb_pca.npy"\ 

# Define variables for all arguments
INPUT_ADATA=$1
OUTPUT_ADATA=$2
OUTPUT_PAIR_ADATA=$3
PAIRED_ADATA=$4
# Run the Python script with the variable inputs
python run_pca_paired.py \
    --input_adata "$INPUT_ADATA" \
    --output_adata "$OUTPUT_ADATA" \
    --output_pair_adata "$OUTPUT_PAIR_ADATA" \
    --paired_adata "$PAIRED_ADATA"