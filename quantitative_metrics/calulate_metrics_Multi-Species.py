import pandas as pd
from metrics import compute_cell_type_matching_metrics,compute_batch_diversity_metrics,compute_balanced_metrics
models = ["cellfishing","pca","scvi","linearscvi","cellblast","scfoundation","scgpt","scimilarity","uce","geneformer","scmulan","cellplm","nicheformer"]
for dataset in ["Cerebellum-dev","Muscle-aging"]:
    for cross in ["human_mouse","mouse_human"]:
        all_result_df = dict()
        for model in models:
            print(model)
            method_df = dict()
            celltype = pd.read_csv(f'./raw_data/{dataset}/{cross}/{model}/cell_type.csv')    
            method = pd.read_csv(f'./raw_data/{dataset}/{cross}/{model}/assay.csv')
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
                print("ct_metrics")
                batch_metrics = compute_batch_diversity_metrics(list(query), celltype.values[:,0:k], method.values[:,0:k],k)
                print("batch_metrics")
                balanced_metrics = compute_balanced_metrics(list(query), celltype.values[:,0:k],k)
                print("balanced_metrics")
                combined_metrics = {**ct_metrics, **batch_metrics, **balanced_metrics}
                method_df['method_top_' + str(k)] = combined_metrics
            method_df_mean = pd.DataFrame(method_df).mean(axis=1)
            all_result_df[model] = method_df_mean
        all_result_df = pd.DataFrame(all_result_df)
        all_result_df.to_csv(f"./metric/result_all_metric_{dataset}_{cross}_l2_norm_top_50.csv")