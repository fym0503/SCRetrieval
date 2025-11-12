#!/usr/bin/env python
"""
Cell Retrieval Pipeline
Supports:
  - Single dataset (split by query/target index)
  - Paired datasets (multiple query + reference pairs)
"""

import scanpy as sc
import numpy as np
import argparse
import faiss
import warnings
import time
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from faiss_retrieval import similarity_search

warnings.filterwarnings('ignore')

# =============================================================================
# Argument Parsing
# =============================================================================
parser = argparse.ArgumentParser(description='Cell retrieval from embeddings.')
parser.add_argument('--input_adata', type=str, nargs='+', required=True,
                    help='Path(s) to input .h5ad file(s) [query dataset(s)]')
parser.add_argument('--input_embeddings', type=str, nargs='+', required=True,
                    help='Path(s) to input .npy embedding file(s) [query embeddings]')
parser.add_argument('--method', type=str, required=True,
                    help='Embedding method name')
parser.add_argument('--retrieved_for_each_cell', type=int, required=True,
                    help='Number of nearest neighbors to retrieve per query cell')
parser.add_argument('--faiss_search', type=str, required=True,
                    choices=['L2', 'cosine'], help='FAISS distance metric')
parser.add_argument('--obs', type=str, required=True,
                    help='Comma-separated obs columns to save (e.g., cell_type,batch)')
parser.add_argument('--query_indexes', type=str, default=None,
                    help='Path to .npy file with query indices (for single-dataset mode)')
parser.add_argument('--target_indexes', type=str, default=None,
                    help='Path to .npy file with target indices (optional)')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory to save results')
parser.add_argument('--has_paired_data', action='store_true',
                    help='Enable paired query-reference mode')
parser.add_argument('--paired_adata', type=str, nargs='*', default=[],
                    help='Path(s) to paired reference .h5ad file(s)')
parser.add_argument('--pair_embeddings', type=str, nargs='*', default=[],
                    help='Path(s) to paired reference .npy embedding file(s)')
parser.add_argument('--norm', type=str, default=None,
                    choices=['standard', 'sc1p', 'no'],
                    help='Normalization: standard (z-score), sc1p (log1p), none')

args = parser.parse_args()

# =============================================================================
# Helper Functions
# =============================================================================
def load_and_normalize(embeddings_list, norm_type):
    """Apply normalization to list of embeddings."""
    normalized = []
    scaler = StandardScaler() if norm_type == 'standard' else None

    for emb in embeddings_list:
        emb = np.array(emb, dtype=np.float32)
        if norm_type == 'sc1p':
            emb = sc.pp.log1p(emb, copy=True)
        elif norm_type == 'standard':
            emb = scaler.fit_transform(emb)
        normalized.append(emb)
    return normalized

def concat_datasets(adata_list, embeddings_list, obs_list):
    """Concatenate AnnData and embeddings, return global indices."""
    all_obs = {}
    for col in obs_list:
        all_obs[col] = np.concatenate([adata.obs[col].values for adata in adata_list])

    all_embeddings = np.concatenate(embeddings_list, axis=0)
    return all_embeddings, all_obs

# =============================================================================
# Main Execution
# =============================================================================
def main():
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Running retrieval: {args.method.upper()}")

    obs_list = [x.strip() for x in args.obs.split(",")]
    norm = args.norm if args.norm != 'no' else None

    # =========================================================================
    # CASE 1: Paired Data Mode (Multiple Query + Reference Datasets)
    # =========================================================================
    if args.has_paired_data:
        if len(args.input_adata) != len(args.input_embeddings):
            raise ValueError("Number of query adata and embeddings must match.")
        if len(args.paired_adata) != len(args.pair_embeddings):
            raise ValueError("Number of reference adata and embeddings must match.")
        

        # Load query datasets
        query_adatas = [sc.read_h5ad(p) for p in args.input_adata]
        query_embs = [np.load(p) for p in args.input_embeddings]

        # Load reference datasets
        ref_adatas = [sc.read_h5ad(p) for p in args.paired_adata]
        ref_embs = [np.load(p) for p in args.pair_embeddings]

        # Normalize
        query_embs = load_and_normalize(query_embs, norm)
        ref_embs = load_and_normalize(ref_embs, norm)

        # Concatenate all
        all_adatas = query_adatas + ref_adatas
        all_embeddings = np.concatenate(query_embs + ref_embs, axis=0)

        # Global index mapping
        n_query = sum(e.shape[0] for e in query_embs)
        query_global_idx = np.arange(n_query)
        ref_global_idx = np.arange(n_query, n_query + sum(e.shape[0] for e in ref_embs))

        # Concatenate obs
        all_obs = {}
        for col in obs_list:
            all_obs[col] = np.concatenate([
                adata.obs[col].values for adata in all_adatas
            ])

    # =========================================================================
    # CASE 2: Single Dataset Mode
    # =========================================================================
    else:
        if len(args.input_adata) > 1 or len(args.input_embeddings) > 1:
            raise ValueError("Single-dataset mode supports only one adata and one embedding.")

        adata = sc.read_h5ad(args.input_adata[0])
        embeddings = np.load(args.input_embeddings[0]).astype(np.float32)
        n_cells = embeddings.shape[0]

        # Load or infer query/target indices
        query_indexes = np.load(args.query_indexes) if args.query_indexes else np.arange(n_cells)
        if args.target_indexes:
            target_indexes = np.load(args.target_indexes)
        else:
            all_idx = np.arange(n_cells)
            target_indexes = np.setdiff1d(all_idx, query_indexes)

        # Normalize
        if norm == 'sc1p':
            embeddings = sc.pp.log1p(embeddings, copy=True)
        elif norm == 'standard':
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)

        # Map to global (same) space
        all_embeddings = embeddings
        query_global_idx = query_indexes
        ref_global_idx = target_indexes

        all_obs = {col: adata.obs[col].values for col in obs_list}

    # =========================================================================
    # Run Retrieval
    # =========================================================================
    print(f"Total cells in search space: {all_embeddings.shape[0]}")
    print(f"Query cells: {len(query_global_idx)}, Reference cells: {len(ref_global_idx)}")

    distances, indices = similarity_search(
        args, all_embeddings, query_global_idx, ref_global_idx
    )

    # =========================================================================
    # Save Results
    # =========================================================================
    k = args.retrieved_for_each_cell
    columns = ['Query'] + [f'Result-{i}' for i in range(1, k + 1)]

    # Save index mapping
    result_index = np.column_stack([query_global_idx.reshape(-1, 1), ref_global_idx[indices]])
    df_index = pd.DataFrame(result_index, columns=columns)
    df_index.to_csv(os.path.join(args.output_dir, "index.csv"), index=False)

    
    np.save(os.path.join(args.output_dir, "distances.npy"), distances)

    # Save annotation CSVs
    for col in obs_list:
        values = all_obs[col]
        df_ann = pd.DataFrame(values[result_index], columns=columns)
        filename = "batch.csv" if col == 'donor_id' else f"{col}.csv"
        df_ann.to_csv(os.path.join(args.output_dir, filename), index=False)

    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
