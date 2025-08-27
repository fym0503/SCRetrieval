
#!/bin/bash

# Example command:
# python ./run_scimilarity.py --input_adata  "../../raw_data/Limb-Embryo/st.h5ad" --output_adata  ../../raw_data/Limb-Embryo/embeddings/st_hindlimb_scimilarity.npy --model_path "./scimilarity/"

# Define variables for all arguments
INPUT_ADATA=$1
OUTPUT_ADATA=$2
MODEL_PATH=$3
# Run the Python script with the variable inputs
python run_scimilarity.py \
    --input_adata "$INPUT_ADATA" \
    --output_adata "$OUTPUT_ADATA" \
    --model_path "$MODEL_PATH" \