import scanpy as sc
import numpy as np
import argparse
import faiss
import warnings
import scanpy as sc
import time
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import random
import numpy as np
from faiss_retrieval import similarity_search
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# Number of components for whitening
N_COMPONENTS = 128

parser = argparse.ArgumentParser(description='Process single-cell data.')
parser.add_argument('--input_adata', type=str, required=True, help='Path to input .h5ad file')
parser.add_argument('--input_embeddings', type=str, required=True, help='Path to input .npy file')
parser.add_argument('--method', type=str, required=True, help='Method used for embeddings; all if you want to run all methods')
parser.add_argument('--retrieved_for_each_cell', type=int, required=True, help='Number of cells to retrieve for each cell')
parser.add_argument('--faiss_search', type=str, required=True, help='Faiss search method')
parser.add_argument('--obs', type=str, required=True)
parser.add_argument('--query_indexes', type=str, default=None)
parser.add_argument('--target_indexes', type=str, default=None)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--has_paired_data', action='store_true')
parser.add_argument('--paired_adata', type=str, default=None, help='Path to input paired .h5ad file')
parser.add_argument('--pair_embeddings', type=str, default=None, help='Path to input paired .npy file')
parser.add_argument('--norm', type=str, default=None)
args = parser.parse_args()

if args.has_paired_data is not None:
    adata = sc.read(args.input_adata)
    pair_adata = sc.read(args.paired_adata)
    obs_list = args.obs.split(",")

    scaler = StandardScaler()
    scaler_paired =  StandardScaler()

    print(f"########### {args.method.capitalize()} ###########")
    embeddings = np.load(args.input_embeddings)
    embeddings_pair = np.load(args.pair_embeddings)

    print("Embeddings shape: ", embeddings.shape)
    query_indexes = np.array([i for i in range(np.shape(embeddings)[0])])

    target_indexes = np.array([i+np.shape(embeddings)[0] for i in range(np.shape(embeddings_pair)[0])])
    # print(target_indexes)
    if(args.norm=="sc1p"):
        embeddings = sc.pp.log1p(embeddings)
        embeddings_pair = sc.pp.log1p(embeddings_pair)
    elif(args.norm=="standard"):
        embeddings = scaler.fit_transform(embeddings)
        embeddings_pair = scaler_paired.fit_transform(embeddings_pair)
    embeddings = np.concatenate((embeddings,embeddings_pair))

    distances, index = similarity_search(args, embeddings, query_indexes, target_indexes)

    columns = ['Query'] + ['Result-' + str(i) for i in range(1,args.retrieved_for_each_cell+1)]
    df = pd.DataFrame(np.concatenate([query_indexes.reshape(-1,1),target_indexes[index]],axis=1),columns=columns)
    os.makedirs(args.output_dir,exist_ok=True)

    df.to_csv(args.output_dir + "/index.csv")
    for i in obs_list:
        temp = list(adata.obs[i])
        temp.extend(list(pair_adata.obs[i]))
        target = pd.DataFrame(np.array(temp)[df.values],columns = columns)
        if i=='donor_id':
            target.to_csv(args.output_dir + "/" + "batch" + ".csv")
        else:
            target.to_csv(args.output_dir + "/" + i + ".csv")
    # temp = list(adata.obs.cell_type)
    # temp.extend(list(pair_adata.obs.cell_type))
    # target = pd.DataFrame(np.array(temp)[df.values],columns = columns)
    # target.to_csv(args.output_dir + "/" + "cell_type" + ".csv")

else:
    adata = sc.read(args.input_adata)
    obs_list = args.obs.split(",")


    print(f"########### {args.method.capitalize()} ###########")
    embeddings = np.load(args.input_embeddings)
    embeddings = np.float32(embeddings)
    print("Embeddings shape: ", embeddings.shape)
    query_indexes = np.load(args.query_indexes)
    if args.target_indexes is not None:
        target_indexes = np.load(args.target_indexes)
    else:
        target_indexes = np.array([i for i in range(len(adata)) if i not in query_indexes])
    # print(embeddings)

    distances, index = similarity_search(args, embeddings, query_indexes, target_indexes)
    print(distances.shape)
    columns = ['Query'] + ['Result-' + str(i) for i in range(1,args.retrieved_for_each_cell+1)]
    df = pd.DataFrame(np.concatenate([query_indexes.reshape(-1,1),target_indexes[index]],axis=1),columns=columns)
    os.makedirs(args.output_dir,exist_ok=True)
    df.to_csv(args.output_dir + "/index.csv")
    for i in obs_list:
        target = pd.DataFrame(np.array(adata.obs[i])[df.values],columns = columns)
        target.to_csv(args.output_dir + "/" + i + ".csv")
    np.save(args.output_dir + "/cosine_similarity.npy",distances)
