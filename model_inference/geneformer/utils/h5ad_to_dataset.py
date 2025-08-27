from geneformer import TranscriptomeTokenizer
import os
import scanpy as sc
from collections import Counter
from geneformer import EmbExtractor
import pandas as pd
import numpy as np
def data_preparation_geneformer(input_adata):
    #tk = TranscriptomeTokenizer({"cell_type": "cell_type", "tissue_general": "organ","donor_id":"donor_id","index":"index"}, nproc=16)
    
    tk = TranscriptomeTokenizer({'cell_type': "celltype","index":"index"}, nproc=16)
    file_name = input_adata.split('/')[-1]
    adata = sc.read(input_adata)
    adata.obs['index'] = list(adata.obs.index)
    #qc = sc.pp.calculate_qc_metrics(adata)
    #adata.obs['n_counts'] = qc[0]['total_counts']
    
    #organ = list(Counter(adata.obs['tissue_general']).keys())[0]
    adata_name = file_name.split('.')[0]
    try: 
        os.mkdir("./model/{}".format(adata_name))
        adata.write("./model/{}/{}".format(adata_name,file_name))   
        tk.tokenize_data("./model/{}".format(adata_name), 
                        "./model/{}".format(adata_name), 
                        "{}".format(adata_name), 
                        file_format="h5ad")
    except OSError as error: 
        print("The h5ad file was already processed")
    
    return "./model/{}/{}.dataset".format(adata_name,adata_name), adata_name,adata
def extract_emb_geneformer(original_index_list,num_classes,output_dir,dataset_path,dataset_organ,output_adata,adata):
    embex = EmbExtractor(model_type="Pretrained",
                     #num_classes=num_classes,
                     #filter_data={"cell_type":["Cardiomyocyte1","Cardiomyocyte2","Cardiomyocyte3"]},
                     max_ncells=1000000,
                     emb_layer=-1,
                     emb_label=["celltype","index"],
                     labels_to_plot=["celltype"],
                     forward_batch_size=10,
                     nproc=10,
                     )
    embs = embex.extract_embs(output_dir,
                            dataset_path,
                            "{}".format(output_dir),
                            "{}".format(dataset_organ))
#     embs = embex.extract_embs("/research/d2/fyp23/khlee0/Geneformer/examples/pretraining_new_model/models/240203_102205_self_prepared_data_corrected_no_freezed_L6_emb256_SL2048_E3_B1_LR0.001_LSlinear_WU10000_Oadamw_DS4/models",
#     "/research/d2/fyp23/khlee0/kenny-scRetrieval-FM/scRetrieval-FM/model/Ciona_intestinalis_larva_dge_corrected/Ciona_intestinalis_larva_dge_corrected.dataset",
#  "/research/d2/fyp23/khlee0/Geneformer/examples/pretraining_new_model/models/240203_102205_self_prepared_data_corrected_no_freezed_L6_emb256_SL2048_E3_B1_LR0.001_LSlinear_WU10000_Oadamw_DS4/models",
# "geneformer_no_freeze")
    # embs = embs.drop(columns=['index'])
    # embex.plot_embs(embs, 
    #            plot_style="umap",
    #            output_directory="/research/d2/fyp23/khlee0/kenny-scRetrieval-FM/scRetrieval-FM/model/Ciona_intestinalis_larva_dge_corrected",  
    #           output_prefix="emb_plot_no_freezed",
    #           max_ncells_to_plot=100000)
    embs = pd.DataFrame(embs)
    print(embs)
    embs['index'] = embs['index'].astype("category")
    embs['index'] = embs['index'].cat.set_categories(original_index_list)
    embs = embs.sort_values(["index"])
    embs.index = original_index_list
    embs.columns = [str(i) for i in embs.columns]
    print(embs)
    #adata.obsm["X_geneformer"] = embs.iloc[:,:-3]
    npy_output = embs.iloc[:,:-3].to_numpy()
    #adata.write(output_adata)
    np.save(output_adata,npy_output)