import warnings
warnings.filterwarnings("ignore")
import argparse
import hdf5plugin
import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix
from CellPLM.utils import set_seed
from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
import scanpy as sc
import matplotlib.pyplot as plt
import time


parser = argparse.ArgumentParser(description='Generating CellPLM embeddings.')
parser.add_argument('--input_adata_path', type=str, required=True, help='Path to input .h5ad file')
parser.add_argument('--output_embeddings_path', type=str, required=True, help='Path to output .npy file')
args = parser.parse_args()

PRETRAIN_VERSION = '20231027_85M'
DEVICE = 'cuda'

set_seed(42)

data = ad.read_h5ad(args.input_adata_path)

start_cpu_time = time.process_time()
data.obs_names_make_unique()
data.obs.columns = list(data.obs.columns)
tmp = list()
print("Cell count: ", data.n_obs)
for i in range(len(data.var.index)):
    tmp.append(data.var.feature_name[data.var.index[i]])
data.var.index = tmp
data.var_names_make_unique()
print(data.var)

pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION, # Specify the pretrain checkpoint to load
                                 pretrain_directory='ckpt')

embedding = pipeline.predict(data, inference_config={'batch_size':1000},device=DEVICE) 

data.obsm['emb'] = embedding.cpu().numpy()

print(data.obsm['emb'])
print(data.obsm['emb'].shape)
execution_time = time.process_time() - start_cpu_time

print(f"Embedding extraction time in seconds: {execution_time:.6f} seconds")
np.save(args.output_embeddings_path, data.obsm['emb'])
