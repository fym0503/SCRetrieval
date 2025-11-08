import anndata
import numpy as np
import scanpy as sc
import pandas as pd
import os


def de_analysis(adata, key, group):
    sc.tl.rank_genes_groups(adata, key, groups=group, method='wilcoxon')
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
    top_genes_scores_pvals = pd.concat(
        [top_genes, top_scores, top_pvals, top_logfoldchanges],
        keys=['genes', 'scores', 'pvals_adj', 'logfoldchanges'], axis=1)
    return top_genes_scores_pvals

model_name_all = ['cellplm', 'scMulan', 'geneformer', 'scgpt', 'scimilarity', 
                  'uce', 'scfoundation', 'scvi', 'linearscvi', 'pca', 'cellblast', 'cellfishing']

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def compute_consistency_with_top_genes(gene_list, top_genes):
    consistency_scores = []
    for genes in gene_list:
        consistency_scores.append(jaccard_similarity(genes, top_genes))
    return consistency_scores

def voting_top_genes(gene_list):
    gene_counts = {}
    for genes in gene_list:
        for gene in genes:
            if gene in gene_counts:
                gene_counts[gene] += 1
            else:
                gene_counts[gene] = 1
  
    sorted_genes = sorted(gene_counts.items(), key=lambda x: x[1], reverse=True)
    top_genes = {gene for gene, count in sorted_genes[:50]}
    return top_genes

path = "/ailab/user/chenpengan/fanyimin/retrieval_clean_codebase/DEG_analysis/output/predicted_deg"
adata = anndata.read("/ailab/user/chenpengan/fanyimin/retrieval/adata_pbmc_benchmark.h5ad")
all_celltype = np.array(adata.obs['CellType'])

for method in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    performance_dict = dict()
    index = pd.read_csv("/ailab/user/chenpengan/fanyimin/retrieval_clean_codebase/DEG_analysis/output/platform_sample/" + str(method) + "_index.csv")
    query_index = list(index['Query'])
    
    for item in range(len(index)):
        performance_dict[item] = []
        gene_list = []
        all_cell_type_item = all_celltype[item]
        
        for model in model_name_all:
            top_genes_scores_pvals = pd.read_csv(path + "/" + model + "_" + str(method) + "_method_l2_norm/de_tests/" + str(query_index[item]) + ".csv")
            top_genes_scores_pvals = np.array(top_genes_scores_pvals['genes'][0:100])
            gene_list.append(set(top_genes_scores_pvals))
        
        top_genes_voted = voting_top_genes(gene_list)
        
        consistency_scores = compute_consistency_with_top_genes(gene_list, top_genes_voted)
        performance_dict[item] = consistency_scores
    
    consistency_df = pd.DataFrame.from_dict(performance_dict, orient='index', columns=model_name_all)
    consistency_df.mean().to_csv("deg_voting/" + str(method) + ".csv")
