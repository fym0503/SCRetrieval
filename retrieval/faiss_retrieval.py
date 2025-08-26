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

# Number of components for whitening
N_COMPONENTS = 128

# Number of clusters 
NLIST = 32
# Number of neighbours searched
N_PROBE = 4

# Use a single GPU
# TODO: implement faiss GPU     
# res = faiss.StandardGpuResources()

def similarity_search(args, embeddings, query_indexes, retrieve_indexes):
    # indexes = []
    # types_unique_set = set()
    # types_unique = np.unique(types)
    # actual = np.empty(shape=[0, 0])
    # predicted = np.empty(shape=[0, 0])
    DIMENSION = embeddings.shape[1]
    # if args.normalization == 'b':
    #     DIMENSION = min(DIMENSION, N_COMPONENTS)
    # # Number of nighbours searched
    # NEIGHBORS = args.retrieved_for_each_cell
    start = time.time()
    if args.faiss_search == 'IP':
        quantizer = faiss.IndexFlatIP(DIMENSION)
        index = faiss.IndexIVFFlat(quantizer,  DIMENSION, NLIST, faiss.METRIC_INNER_PRODUCT)
    if args.faiss_search == 'L2':
        quantizer = faiss.IndexFlatL2(DIMENSION)
        index = faiss.IndexIVFFlat(quantizer, DIMENSION, NLIST,faiss.METRIC_INNER_PRODUCT)
    # print(embeddings[retrieve_indexes])
    faiss.normalize_L2(embeddings)
    # print(retrieve_indexes)
    index.train(embeddings[retrieve_indexes])
    index.add(embeddings[retrieve_indexes])

    print('Building index tree: ', time.time() - start)
    # Number of clusters to be examined
    index.nprobe = N_PROBE
    start = time.time()
    # GOING THROUGH QUERY CELLS
    D, I = index.search(embeddings[query_indexes], args.retrieved_for_each_cell)
    
    return D, I
