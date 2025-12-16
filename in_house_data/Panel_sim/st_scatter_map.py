import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import numpy as np
import anndata
from sklearn.metrics.pairwise import cosine_similarity

adata = sc.read("../shared_data/X5704ADHC.h5ad")
adata_ref = anndata.read("/workspace/fuchenghao/retrieval_neuron/combined_reference.h5ad")

healthy_mask = adata_ref.obs['Disease'] == 'NO'
healthy_ref_indices = set(np.where(healthy_mask)[0])

models = ['pca_st', 'linearscvi_st', 'scvi_st', 'cellblast_st', 'scgpt', 'scimilarity', 'uce', 'cellplm', 'geneformer', 'scfoundation_new', 'scmulan']

for model in models:
    print(f"Processing model: {model}")
    
    obs_df = adata.obs.copy().reset_index()
    index_df = pd.read_csv(f"../shared_data/st_reference/{model}/index.csv")
    reference_embedding = np.load(f"../shared_data/reference_embedding/{model}/embeds.npy")
    query_embedding = np.load(f"../shared_data/st_in_house_embedding/{model}/embeds.npy")
    
    concatenated_embedding = np.concatenate([query_embedding, reference_embedding], axis=0)
    query_len = len(query_embedding)
    
    queries = index_df['Query'].values
    results_cols = [f'Result-{j}' for j in range(1, 51)]
    ref_indices = index_df[results_cols].values
    
    average_similarities = np.full(len(queries), np.nan)
    
    valid_mask = ~pd.isna(ref_indices)
    adjusted_refs = (ref_indices - query_len).astype(int)
    
    for i in range(len(queries)):
        valid_refs = adjusted_refs[i][valid_mask[i]]
        if len(valid_refs) == 0:
            continue
        
        healthy_refs = [r for r in valid_refs if r in healthy_ref_indices]
        
        if len(healthy_refs) == 0:
            continue
        
        healthy_refs_original = np.array(healthy_refs) + query_len
        sims = cosine_similarity(
            concatenated_embedding[queries[i]:queries[i]+1],
            concatenated_embedding[healthy_refs_original]
        )[0]
        
        average_similarities[i] = np.mean(sims)
    
    obs_df['average_similarity'] = average_similarities
    
    valid_sims = average_similarities[~np.isnan(average_similarities)]
    lower = np.percentile(valid_sims, 2.5)
    upper = np.percentile(valid_sims, 97.5)
    
    filter_mask = (obs_df['average_similarity'] >= lower) & (obs_df['average_similarity'] <= upper)
    obs_df_filtered = obs_df[filter_mask]
    
    print(f"Original points: {len(obs_df)}, Filtered points: {len(obs_df_filtered)}")
    print(f"Similarity range (5%-95%): {lower:.3f} - {upper:.3f}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(
        np.array(obs_df_filtered['center_x']),
        np.array(obs_df_filtered['center_y']),
        c=obs_df_filtered['average_similarity'],
        cmap='RdYlBu_r',
        s=2,
        rasterized=True
    )
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    cb = plt.colorbar(scatter, ax=ax)
    cb.ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(f'./plot_sim/{model}_st_similarity_heatmap.pdf', 
                bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Saved: ./plot_sim/{model}_st_similarity_heatmap.pdf")
    print("-" * 50)