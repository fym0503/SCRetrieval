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
parser.add_argument("--number")

args = parser.parse_args()
scimilarity_path = "novel_cell_type/ioncyte"

y_score_all = []



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
    'lung','bronchus']

for platform in platforms:
    try:
        average_precision_dict[platform] = dict()
        auc_roc_dict[platform] = dict()
        y_score_all = []
        for model in models:
            seen = np.mean(np.load(os.path.join(scimilarity_path,f"{platform}/{model}","not_novel_cosine_similarity.npy"))[:,0:int(args.number)],axis=1)
            unseen = np.mean(np.load(os.path.join(scimilarity_path,f"{platform}/{model}","novel_cosine_similarity.npy"))[:,0:int(args.number)],axis=1)
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

scimilarity_path =  "novel_cell_type/ioncyte/reverse"

average_precision_dict = dict()
auc_roc_dict = dict()

platforms = [
    'lung','bronchus']

for platform in platforms:
    try:
        average_precision_dict[platform] = dict()
        auc_roc_dict[platform] = dict()
        y_score_all = []
        for model in models:
            seen = np.mean(np.load(os.path.join(scimilarity_path,f"{platform}/{model}","not_novel_cosine_similarity.npy"))[:,0:int(args.number)],axis=1)
            unseen = np.mean(np.load(os.path.join(scimilarity_path,f"{platform}/{model}","novel_cosine_similarity.npy"))[:,0:int(args.number)],axis=1)
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

df_reverse = pd.DataFrame(auc_roc)



df_new = pd.concat([df,df_reverse],axis=1)
row_means = df_new.mean(axis=1)
df_new['mean'] = row_means
# df1_sorted = df_new.reindex(row_means.sort_values(ascending=False).index)
# df1_sorted['mean'] = list(row_means.sort_values(ascending=False))
df_new.to_csv("all_data_" + str(args.number) + ".csv")
