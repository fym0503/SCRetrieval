import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import pandas as pd
import argparse
from matplotlib import cm

# 设置命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument("--cell_type")
parser.add_argument("--number",type=int,default=1)

args = parser.parse_args()
scimilarity_path = "novel/disease_novel"

y_score_all = []

df = pd.read_csv(scimilarity_path + "/scgpt/novel_cell_type.csv")
cell_types = np.array(df['Query'])

# models = ["cellblast", "cellplm", "geneformer", "pca", "scgpt", "scimilarity", "scmulan", "uce", 'scfoundation', 'scvi', 'linearscvi']
models = ['pca','scvi','linearscvi','cellblast',"cellplm", "geneformer","scgpt", "scimilarity", "scmulan", "uce", 'scfoundation','cellfishing']
name_dict = {
    "cellblast": "CellBlast",
    "cellplm": "CellPLM",
    "geneformer": "Geneformer",
    "pca": "PCA",
    "scgpt": "scGPT",
    "scimilarity": "SCimilarity",
    'scmulan': 'scMuLan',
    'uce': "UCE",
    'scfoundation': 'scFoundation',
    'scvi': 'scVI',
    'linearscvi': 'LDVAE',
    'cellfishing':"CellFishing.jl"

}

for model in models:
    seen = np.mean(np.load(os.path.join(scimilarity_path, f"{model}", "not_novel_cosine_similarity.npy"))[:, 0:args.number],axis=1)

    unseen = np.mean(np.load(os.path.join(scimilarity_path, f"{model}", "novel_cosine_similarity.npy"))[:, 0:args.number],axis=1)

    if model == 'cellfishing':
        seen = 1 / seen
        unseen = 1 / unseen

    seen_df = pd.read_csv(os.path.join(scimilarity_path, "scgpt", "not_novel_cell_type.csv"))
    seen_index = []
    
    for i in range(len(seen_df['Query'])):
        if args.cell_type.split("_")[0] in seen_df['Query'][i]:
            seen_index.append(i)
        
    index = np.where(cell_types == args.cell_type)[0]

    seen_flat = seen.flatten()[seen_index]
    unseen_flat = unseen.flatten()[index]
    y_true = np.append(np.zeros(len(unseen_flat)), np.ones(len(seen_flat)))
    y_score = np.append(unseen_flat, seen_flat)
    y_score_all.append(y_score)

np.random.seed(42)  # For reproducibility

models_dict = dict()
for i in range(len(models)):
    models_dict[name_dict[models[i]]] = y_score_all[i]


roc_auc_all = []

for idx, (model_name, y_scores) in enumerate(models_dict.items()):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    roc_auc_all.append(roc_auc)
    

df = pd.DataFrame(roc_auc_all)
df.index = list(models_dict.keys())
os.makedirs("scores_" + str(args.number) + "/", exist_ok=True)
df.to_csv("scores_" + str(args.number) + "/" + args.cell_type + ".csv")