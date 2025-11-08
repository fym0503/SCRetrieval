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
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score,accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--cell_type")
parser.add_argument("--number",type=int,default=1)
parser.add_argument("--threshold",type=float,default=0.92)
args = parser.parse_args()


if __name__ == '__main__':

    scimilarity_path = "/Users/liuxinyuan/Desktop/new_projects/cell_retrieval/result_base/disease_novel"

    y_score_all = []

    # df = pd.read_csv(scimilarity_path + "/scgpt/novel_cell_type.csv")
    # cell_types = np.array(df['Query'])

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

    df = pd.read_csv(scimilarity_path + "/scgpt/novel_cell_type.csv")
    cell_types = np.array(df['Query'])

    average_precision_dict = dict()
    precision_dict = dict()
    accuracy_dict = dict()
    recall_dict = dict()
    f1_dict = dict()

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
        y_true = np.append(np.ones(len(unseen_flat)), np.zeros(len(seen_flat)))
        y_score = np.append(unseen_flat, seen_flat)
        y_score_all.append(y_score)

    np.random.seed(42)  # For reproducibility
    models_dict = dict()
    for i in range(len(models)):
        models_dict[name_dict[models[i]]] = y_score_all[i]

    for model_name, y_scores in models_dict.items():
        precision_dict[model_name]=precision_score(y_true=y_true, y_pred=y_scores<args.threshold)
        recall_dict[model_name] = recall_score(y_true=y_true, y_pred=y_scores<args.threshold)
        f1_dict[model_name] = f1_score(y_true=y_true, y_pred=y_scores<args.threshold)
        accuracy_dict[model_name] = accuracy_score(y_true=y_true, y_pred=y_scores<args.threshold)
        # average_precision_dict[platform][model_name] = average_precision_score(y_true, y_scores)
        # auc_roc_dict[model_name] = roc_auc_score(y_true=y_true, y_pred=y_scores)
            

    precision_df = pd.DataFrame(precision_dict.values())
    precision_df.index = list(precision_dict.keys())
    os.makedirs("precision_" + str(args.number) + "_" + str(args.threshold) + "/",exist_ok=True)
    precision_df.to_csv("precision_" + str(args.number) + "_" + str(args.threshold) + "/" + args.cell_type + ".csv")
            

    accuracy_df = pd.DataFrame(accuracy_dict.values())
    accuracy_df.index = list(accuracy_dict.keys())
    os.makedirs("accuracy_" + str(args.number) + "_" + str(args.threshold) + "/",exist_ok=True)
    accuracy_df.to_csv("accuracy_" + str(args.number) + "_" + str(args.threshold) + "/" + args.cell_type + ".csv")

    recall_df = pd.DataFrame(recall_dict.values())
    recall_df.index = list(recall_dict.keys())
    os.makedirs("recall_" + str(args.number) + "_" + str(args.threshold) + "/",exist_ok=True)
    recall_df.to_csv("recall_" + str(args.number) + "_" + str(args.threshold) + "/" + args.cell_type + ".csv")

    f1_df = pd.DataFrame(f1_dict.values())
    f1_df.index = list(f1_dict.keys())
    os.makedirs("f1_" + str(args.number) + "_" + str(args.threshold) + "/",exist_ok=True)
    f1_df.to_csv("f1_" + str(args.number) + "_" + str(args.threshold) + "/" + args.cell_type + ".csv")