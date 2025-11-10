import anndata
import numpy as np
import scanpy as sc
import pandas as pd
import os
import time
import argparse



def de_analysis(adata, key,group):
    sc.tl.rank_genes_groups(adata, key,groups=group,method='wilcoxon')
    rank_genes_groups = adata.uns['rank_genes_groups']
    groups = rank_genes_groups['names'].dtype.names

    top_genes = pd.DataFrame({group: rank_genes_groups['names'][group] for group in groups})
    top_scores = pd.DataFrame({group: rank_genes_groups['scores'][group] for group in groups})

    top_pvals = pd.DataFrame(
        {group: rank_genes_groups['pvals_adj'][group] for group in groups}
    )

    top_logfoldchanges = pd.DataFrame(
        {group: rank_genes_groups['logfoldchanges'][group] for group in groups}
    )
    top_genes_scores_pvals = pd.concat([top_genes, top_scores, top_pvals, top_logfoldchanges], keys=['genes', 'scores', 'pvals_adj','logfoldchanges'], axis=1)
    return top_genes_scores_pvals

model_name_all = ['cellplm','scMulan','geneformer','scgpt','scimilarity','uce','scfoundation','scvi','linearscvi','pca','cellblast','cellfishing']


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0
    return intersection / union

def similarity_heatmap(gene_list):
    sim_heatmap = np.zeros((len(gene_list), len(gene_list)))
    for i in range(len(gene_list)):
        for j in range(len(gene_list)):
            sim_heatmap[i][j] = jaccard_similarity(set(gene_list[i]), set(gene_list[j]))
    return sim_heatmap

def compute_jaccard_novelty(model_outputs):
    novelty_scores = []
    for i, output in enumerate(model_outputs):
        # Union of all other models' outputs
        other_outputs = set().union(*[model_outputs[j] for j in range(len(model_outputs)) if j != i])
        all_outputs = set().union(*[model_outputs[j] for j in range(len(model_outputs))])
        # Compute intersection and novelty score
        intersection = output.intersection(other_outputs)
        novelty_score = 1 - len(intersection) / len(output) if len(output) > 0 else 0
        novelty_scores.append(novelty_score)
    
    return novelty_scores

path = "downstream_output/DEG_analysis/output/predicted_deg"
adata = anndata.read("datasets/adata_pbmc_benchmark.h5ad")
overlap_map = np.zeros((len(model_name_all), len(model_name_all)))
all_celltype = np.array(adata.obs['CellType'])


for method in [1,2,3,4,5,6,7,8,9]:
    performance_dict = dict()
    index = pd.read_csv("downstream_output/DEG_analysis/output/platform_sample/" + str(method) + "_index.csv")
    query_index = list(index['Query'])
    for item in range(len(index)):
        performance_dict[item] = []
        gene_list = []
        all_cell_type_item = all_celltype[item]
        for model in model_name_all:
            top_genes_scores_pvals = pd.read_csv(path + "/" + model + "_" + str(method) + "_method_l2_norm/de_tests/" + str(query_index[item]) + ".csv")
            top_genes_scores_pvals = np.array(top_genes_scores_pvals['genes'][0:100])
            gene_list.append(set(top_genes_scores_pvals))
        
        sim_heatmap = similarity_heatmap(gene_list)
        overlap_map += sim_heatmap
    overlap_map = overlap_map / len(index)
    df = pd.DataFrame(overlap_map)
    df.index = model_name_all
    df.columns = model_name_all
    os.makedirs("deg_overlap",exist_ok=True)
    df.to_csv("deg_overlap/" + str(method) + ".csv")
