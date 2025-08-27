

#!/bin/bash

# Example command:
# python ./run_scgpt.py --input_adata "../../raw_data/Limb-Embryo/st.h5ad" --output_path "../../raw_data/Limb-Embryo/embeddings/st_hindlimb_scgpt.npy" --model_path "./scGPT/model_weight/human"


# Define variables for all arguments
INPUT_ADATA=$1
OUTPUT_PATH=$2
MODEL_PATH=$3
# Run the Python script with the variable inputs
python run_scgpt.py \
    --input_adata "$INPUT_ADATA" \
    --output_path "$OUTPUT_PATH" \
    --model_path "$MODEL_PATH" \