import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" ## set your available devices, each use ~2G GPU-MEMORY
import scanpy as sc
import numpy as np
import scMulan
from scMulan import GeneSymbolUniform
import argparse
import time

# Original loop-based approach:
# paths=['../../raw_data/Limb-Embryo/st.h5ad','../../raw_data/Limb-Embryo/sc.h5ad']
# for data_path in paths:

# Define command-line argument parser
parser = argparse.ArgumentParser(description="Process single-cell data and extract embeddings.")
parser.add_argument('--data_path', type=str, required=True, help="Path to the input .h5ad file")
parser.add_argument('--output_dir', type=str, required=True, help="Path to the output .h5ad file")
args = parser.parse_args()

# Use the provided data_path
data_path = args.data_path
parts = data_path.split('/')
name = parts[-1].split('.')[0]

adata = sc.read(data_path)
# adata = adata[:10000]
# print(adata.obs.columns)
# print(adata.obs)
# print(adata.obsm)
# print(adata.var)
# print(adata.uns)

# print(adata.var.index)
adata.var_names = adata.var.feature_name
# query_gene_list = np.array(adata.var_names)
# print(query_gene_list)
adata_GS_uniformed = GeneSymbolUniform(input_adata=adata,
                                output_dir="Data/",
                                output_prefix=name)

# if adata_GS_uniformed.X.max() > 10:
sc.pp.normalize_total(adata_GS_uniformed, target_sum=1e4) 
sc.pp.log1p(adata_GS_uniformed)

ckp_path = 'ckpt/ckpt_scMulan.pt'

scml = scMulan.model_inference(ckp_path, adata_GS_uniformed)

# scml = scMulan.model_inference(ckp_path, adata)
base_process = scml.cuda_count()
start_cpu_time = time.process_time()
scml.get_cell_types_and_embds_for_adata(parallel=False) #set parallel = False to use single GPU
execution_time = time.process_time() - start_cpu_time
print(f"Embedding extraction time in seconds: {execution_time:.6f} seconds")
adata_embds = scml.adata.obsm['X_scMulan'].copy()
# adata_embds.obs = scml.adata.obs

np.save(args.output_dir, adata_embds)