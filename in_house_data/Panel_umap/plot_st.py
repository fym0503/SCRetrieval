import pandas as pd
import scanpy as sc
import numpy as np
import anndata
import matplotlib.pyplot as plt
from collections import Counter
import os
plt.rcParams['font.family'] = 'Arial'

os.makedirs('./plot_st_umap', exist_ok=True)

models = ['pca_st', 'linearscvi_st', 'scvi_st', 'cellblast_st', 'scgpt', 'scimilarity', 'uce', 'cellplm', 'geneformer', 'scfoundation_new', 'scmulan']

celltype_to_color = {
    "Neuron": "#1f77b4",
    "Astrocyte": "#ff7f0e", 
    "Microglia": "#2ca02c",
    "Oligodendrocyte": "#d62728",
    "Endothelial": "#9467bd",
    "OPC": "#8c564b",
    "Pericyte": "#e377c2",
    "Fibroblast": "#7f7f7f",
    "Lymphocyte": "#bcbd22",
    "Macrophage": "#17becf"
}

def vote(vals):
    cnt = Counter(vals)
    m = max(cnt.values())
    ties = {k for k, v in cnt.items() if v == m}
    for v in vals:
        if v in ties:
            return v

def plot(adata_obj, color, out_path, test='right1'):
    adata_obj.obs[color] = adata_obj.obs[color].astype('category')
    adata_obj.obs[color] = adata_obj.obs[color].cat.reorder_categories(sorted(adata_obj.obs[color].cat.categories))
    adata_obj = adata_obj[adata_obj.obs['pred_cell_type'] != 'Fibroblast'].copy()
    adata_obj = adata_obj[adata_obj.obs['pred_cell_type'] != 'Lymphocyte'].copy()

    colors = [celltype_to_color.get(ct, "#cccccc") for ct in adata_obj.obs[color].cat.categories]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    if test != 'right':
        sc.pl.umap(adata_obj, color=color, ax=ax, show=False, legend_loc=None, palette=colors)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))

        sc.pl.umap(adata_obj, color=color, ax=ax, show=False, legend_loc='right', palette=colors)
                
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    for a in ax.collections:
        a.set_rasterized(True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

for m in models:
    print(m)
    adata = anndata.read_h5ad("../shared_data/X5704ADHC.h5ad")
    embeds = np.load("../shared_data/umap_coords_st/" + m + ".npy")
    index_df = pd.read_csv("../shared_data/st_reference/" + m + "/cell_type.csv")
    
    result_cols = [c for c in index_df.columns if c.startswith("Result-")]
    pred = index_df[result_cols].apply(lambda r: vote(r.values.tolist()), axis=1)
    
    adata_new = anndata.AnnData(X=embeds)
    adata_new.obsm['X_umap'] = embeds
    adata_new.obs = adata.obs.copy()
    adata_new.obs["pred_cell_type"] = pred.values
    
    plot(adata_new, "celltype", "./plot_st_umap/umap_real_" + m + "celltype.pdf")
    plot(adata_new, "pred_cell_type", "./plot_st_umap/umap_pred_" + m + "_celltype.pdf")
    
    plot(adata_new, "pred_cell_type", "./plot_st_umap/umap_pred_" + m + "_celltype1.pdf",test='right')
