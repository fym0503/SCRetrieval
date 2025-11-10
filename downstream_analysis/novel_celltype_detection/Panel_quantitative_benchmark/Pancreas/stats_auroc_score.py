# /ailab/group/pjlab-ai4s/aim/fanyimin
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve,roc_auc_score,roc_curve

from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cell_type")
parser.add_argument("--number")

args = parser.parse_args()

scimilarity_path = "novel/novel_discovery_pancreas"

y_score_all = []

# df = pd.read_csv(scimilarity_path + "/scgpt/novel_cell_type.csv")
# cell_types = np.array(df['Query'])

models = ['pca','scvi','linearscvi','cellblast',"cellplm", "geneformer","scgpt", "scimilarity", "scmulan", "uce", 'scfoundation','cellfishing']
name_dict = {
    "cellblast":"CellBlast",
    "cellplm":"CellPLM",
    "geneformer":"Geneformer",
    "pca":"PCA",
    "scgpt":"scGPT",
    "scimilarity":"SCimilarity",
    'scmulan':'scMuLan',
    'uce':"UCE",
    'scfoundation':'scFoundation',
    'scvi':'scVI',
    'linearscvi':'LDVAE',
    'cellfishing':"CellFishing.jl"
}

df = {"model":[],"aucprc":[]}
df_roc = {"model":[],"aucroc":[]}

average_precision_dict = dict()
auc_roc_dict = dict()

platforms = [
    'smartseq2','celseq2','fluidigmc1','celseq'
]
for platform in platforms:
    try:
        average_precision_dict[platform] = dict()
        auc_roc_dict[platform] = dict()
        y_score_all = []
        for model in models:
            seen = np.mean(np.load(os.path.join(scimilarity_path,f"{model}_{platform}_{args.cell_type}","seen_cosine_similarity.npy"))[:,0:int(args.number)],axis=1)
            unseen = np.mean(np.load(os.path.join(scimilarity_path,f"{model}_{platform}_{args.cell_type}","unseen_cosine_similarity.npy"))[:,0:int(args.number)],axis=1)
            
            seen_flat = seen.flatten()
            unseen_flat = unseen.flatten()
            
            if model == 'cellfishing':
                seen = 1 / seen
                unseen = 1 / unseen
                
            y_true = np.append(np.zeros(len(unseen_flat)),np.ones(len(seen_flat)))
            y_score = np.append(unseen_flat,seen_flat)
            y_score_all.append(y_score)

        np.random.seed(42)  # For reproducibility
        models_dict = dict()
        for i in range(len(models)):
            models_dict[name_dict[models[i]]] = y_score_all[i]

        for model_name, y_scores in models_dict.items():
            average_precision_dict[platform][model_name] = average_precision_score(y_true, y_scores)
            auc_roc_dict[platform][model_name] = roc_auc_score(y_true, y_scores)
    except:
        pass

to_del = []
for i in auc_roc_dict.keys():
    if len(auc_roc_dict[i]) == 0:
        to_del.append(i)

new_auc_roc_dict = dict()
for i in auc_roc_dict.keys():
    if i not in to_del:
        new_auc_roc_dict[i] = auc_roc_dict[i]
        

auc_roc = pd.DataFrame(new_auc_roc_dict)

df = pd.DataFrame(auc_roc)
os.makedirs("scores_" + str(args.number) + "/",exist_ok=True)
df.to_csv("scores_" + str(args.number) + "/" + args.cell_type + ".csv")