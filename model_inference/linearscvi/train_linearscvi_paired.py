import scanpy as sc
import scvi
import os
import argparse
import tempfile
import pickle
import numpy as np


def parse_args():
    parse = argparse.ArgumentParser(description='scvi model training')  
    parse.add_argument('--input_adata', default=None, type=str, help='Input file path') 
    parse.add_argument('--query_index', default=None,  help='query index') 
    parse.add_argument('--output_data', default=None) 
    parse.add_argument('--output_dir', default=None) 
    parse.add_argument('--pair_adata', default=None) 
    parse.add_argument('--output_pair_data', default=None)
    args = parse.parse_args()
    return args

args = parse_args()
scvi.settings.seed = 0
sc.set_figure_params(figsize=(4, 4))
adata_path = args.input_adata
adata = sc.read(
    adata_path
)
pair_adata_path = args.pair_adata

pair_adata = sc.read(
    pair_adata_path
)
if args.query_index is not None:
    query_indexes = np.load(args.query_index)
    train_index = np.array([i for i in range(len(adata)) if i not in query_indexes])

    training_adata = adata[train_index]
else:
    print("no cell filtered")
    training_adata = adata
training_adata.layers['counts'] = training_adata.X.copy()
sc.pp.normalize_total(training_adata)
sc.pp.log1p(training_adata)
training_adata.raw = training_adata
scvi.model.LinearSCVI.setup_anndata(training_adata, layer="counts")
model =  scvi.model.LinearSCVI(training_adata)
model.train()
adata.layers['counts'] = adata.X.copy()
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
adata.raw = adata




npy_output = model.get_latent_representation(adata)
print("embedding was saved in {}".format(args.output_data))
np.save(args.output_data ,npy_output)
pair_adata.layers['counts'] = pair_adata.X.copy()
sc.pp.normalize_total(pair_adata)
sc.pp.log1p(pair_adata)
pair_adata.raw = pair_adata

npy_output = model.get_latent_representation(pair_adata)
np.save(args.output_pair_data ,npy_output)