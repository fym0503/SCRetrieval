import pandas as pd
from metrics import compute_cell_type_matching_metrics,compute_batch_diversity_metrics,compute_balanced_metrics
models = ["cellfishing","pca","scvi","linearscvi","cellblast","scfoundation","scgpt","scimilarity","uce","geneformer","scmulan","cellplm","nicheformer"]
all_result_df = dict()
for model in models:
    method_df = dict()
    for i in range(1,5):
        # print(i)
        if model == "cellfishing":
            celltype = pd.read_csv(f'./raw_data/Pancreas/{model}_{i}_method/celltype.csv')    
            method = pd.read_csv(f'./raw_data/Pancreas/{model}_{i}_method/tech.csv')
        else:
            celltype = pd.read_csv(f'./raw_data/Pancreas/{model}_{i}_method_l2_norm/celltype.csv')    
            method = pd.read_csv(f'./raw_data/Pancreas/{model}_{i}_method_l2_norm/tech.csv')

        query = celltype['Query']
        query_method = method['Query']
        del celltype['Query']
        if 'Unnamed: 0' in celltype.columns:
            del celltype['Unnamed: 0']
        del method['Query']
        if 'Unnamed: 0' in method.columns:
            del method['Unnamed: 0']
        
        for k in [50]:
            ct_metrics = compute_cell_type_matching_metrics(list(query), celltype.values[:,0:k],k)
            batch_metrics = compute_batch_diversity_metrics(list(query), celltype.values[:,0:k], method.values[:,0:k],k)
            balanced_metrics = compute_balanced_metrics(list(query), celltype.values[:,0:k],k)
            combined_metrics = {**ct_metrics, **batch_metrics, **balanced_metrics}
            method_df['method_' + str(i) + '_top_' + str(k)] = combined_metrics
    method_df_mean = pd.DataFrame(method_df).mean(axis=1)
    all_result_df[model] = method_df_mean
all_result_df = pd.DataFrame(all_result_df)
all_result_df.to_csv(f"./metric/result_all_metric_pancreas_l2_norm_top_{k}.csv")
