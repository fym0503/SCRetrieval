from pathlib import Path
import pandas as pd
Path('./metric').mkdir(exist_ok=True)
from metrics import compute_cell_type_matching_metrics,compute_batch_diversity_metrics,compute_balanced_metrics,load_reference_metadata
models = ["cellfishing","pca","scvi","linearscvi","cellblast","scfoundation","scgpt","scimilarity","uce","geneformer","scMulan","cellplm","nicheformer"]
all_result_df = dict()
for model in models:
    method_df = dict()
    for i in range(1,10):
        print(i)
        if model == "cellfishing":
            celltype = pd.read_csv(f'../data/PBMC-Ding/{model}_{i}_method/CellType.csv')    
            method = pd.read_csv(f'../data/PBMC-Ding/{model}_{i}_method/Method.csv')
            reference_labels, reference_batches = load_reference_metadata(
                Path(f'../data/PBMC-Ding/{model}_{i}_method/Method.csv').with_name('reference_obs.csv'), 'CellType', 'Method')
        else:
            celltype = pd.read_csv(f'../data/PBMC-Ding/{model}_{i}_method_l2_norm/CellType.csv')    
            method = pd.read_csv(f'../data/PBMC-Ding/{model}_{i}_method_l2_norm/Method.csv')
            reference_labels, reference_batches = load_reference_metadata(
                Path(f'../data/PBMC-Ding/{model}_{i}_method_l2_norm/Method.csv').with_name('reference_obs.csv'), 'CellType', 'Method')

        query = celltype['Query']
        query_method = method['Query']
        del celltype['Query']
        if 'Unnamed: 0' in celltype.columns:
            del celltype['Unnamed: 0']
        del method['Query']
        if 'Unnamed: 0' in method.columns:
            del method['Unnamed: 0']
        
        for k in [50]:
            ct_metrics = compute_cell_type_matching_metrics(list(query), celltype.values[:,0:k],k, reference_labels=reference_labels)
            batch_metrics = compute_batch_diversity_metrics(list(query), celltype.values[:,0:k], method.values[:,0:k], alpha=0.5, k=k, reference_labels=reference_labels, reference_batches=reference_batches)
            balanced_metrics = compute_balanced_metrics(list(query), celltype.values[:,0:k],k, reference_labels=reference_labels)
            combined_metrics = {**ct_metrics, **batch_metrics, **balanced_metrics}
            method_df['method_' + str(i) + '_top_' + str(k)] = combined_metrics
    method_df_mean = pd.DataFrame(method_df).mean(axis=1)
    all_result_df[model] = method_df_mean
all_result_df = pd.DataFrame(all_result_df)
all_result_df.to_csv(f"./metric/result_all_metric_pbmc_l2_norm_top_{k}.csv")
