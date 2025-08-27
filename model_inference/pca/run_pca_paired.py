import scanpy as sc
import os
import argparse
import tempfile
import pickle
import numpy as np
from sklearn.decomposition import PCA

def parse_args():
    parse = argparse.ArgumentParser(description='Cell Embedding Retrieval from data')  # 2、创建参数对象
    parse.add_argument('--input_adata', default=None, type=str, help='Input file path')  # 3、往参数对象添加参数
    parse.add_argument('--output_adata', default=None, type=str, help='Output file directory')
    parse.add_argument('--output_pair_adata', default=None, type=str, help='Output file directory')
    parse.add_argument('--paired_adata',default=None)
    args = parse.parse_args()  # 4、解析参数对象获得解析对象
    return args

args = parse_args()
# scvi.settings.seed = 0
# sc.set_figure_params(figsize=(4, 4))
#experiment_name = "random_12000_2000_cell_embed_svci_heart_second_time"
#print("Last run with scvi-tools version:", scvi.__version__)

adata_path = args.input_adata
adata = sc.read(
    adata_path
)
paired_adata = sc.read(args.paired_adata)
adata = adata.to_df()
adata.columns = list(adata.columns)
paired_adata = paired_adata.to_df()
paired_adata.columns = list(paired_adata.columns)
adata.columns = list(adata.columns)
paired_adata.columns = list(paired_adata.columns)

# Find overlapping columns
overlap_columns = adata.columns.intersection(paired_adata.columns)

# Keep only overlapping columns in both datasets
adata = adata[overlap_columns]
paired_adata = paired_adata[overlap_columns]

# Group by columns and sum (if needed)
adata = adata.groupby(by=adata.columns, axis=1).sum()
paired_adata = paired_adata.groupby(by=paired_adata.columns, axis=1).sum()

pca = PCA(n_components=30,svd_solver="arpack")
pca.fit(adata.values)
npy_output=pca.transform(adata.values)
np.save(args.output_adata ,npy_output)
npy_output=pca.transform(paired_adata.values)
np.save(args.output_pair_adata ,npy_output)