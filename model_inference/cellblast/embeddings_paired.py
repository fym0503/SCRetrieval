import time
import warnings
import anndata
import Cell_BLAST as cb
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
warnings.simplefilter("ignore")
cb.config.RANDOM_SEED = 0
def parse_args():
    parse = argparse.ArgumentParser(description='Cell Embedding Retrieval from data')  # 2、创建参数对象
    parse.add_argument('--input_adata', default=None, type=str, help='Input file path')  # 3、往参数对象添加参数
    parse.add_argument('--output_adata', default=None, type=str, help='Output file directory')
    parse.add_argument('--output_pair_adata', default=None, type=str, help='Output file directory')
    parse.add_argument('--paired_adata',default=None)
    args = parse.parse_args()  # 4、解析参数对象获得解析对象
    return args

args = parse_args()


adata = anndata.read_h5ad(args.input_adata)
paired_adata = anndata.read_h5ad(args.paired_adata)
paired_adata.var_names = [i for i in paired_adata.var.feature_name]
adata.var_names = [i for i in adata.var.feature_name]
temp_df = adata.to_df()
temp_df.columns = list(temp_df.columns)
temp_df = temp_df.groupby(by=temp_df.columns, axis=1).sum()
adata = sc.AnnData(temp_df)
adata.obs_names = temp_df.index
adata.var_names = temp_df.columns
model = cb.directi.fit_DIRECTi(
        adata,
        latent_dim=10, cat_dim=20, epoch=20, learning_rate=0.001
    )
adata.obsm["X_latent"] = model.inference(adata)
embeddings = adata.obsm["X_latent"]
np.save(args.output_adata , embeddings)
temp_df = paired_adata.to_df()
temp_df.columns = list(temp_df.columns)
temp_df = temp_df.groupby(by=temp_df.columns, axis=1).sum()
paired_adata = sc.AnnData(temp_df)
paired_adata.obs_names = temp_df.index
paired_adata.var_names = temp_df.columns
paired_adata.obsm["X_latent"] = model.inference(paired_adata)
embeddings = paired_adata.obsm["X_latent"]
np.save(args.output_pair_adata , embeddings)
