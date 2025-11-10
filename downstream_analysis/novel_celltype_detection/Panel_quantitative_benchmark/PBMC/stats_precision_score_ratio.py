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

    scimilarity_path = "novel_cell_type/model_seen_unseen_result"

    y_score_all = []

    # df = pd.read_csv(scimilarity_path + "/scgpt/novel_cell_type.csv")
    # cell_types = np.array(df['Query'])

    models = ['pca','scvi','linearscvi','cellblast',"cellplm", "geneformer","scgpt", "scimilarity", "scMulan", "uce", 'scfoundation','cellfishing']
    name_dict = {
        "cellblast":"CellBlast",
        "cellplm":"CellPLM",
        "geneformer":"Geneformer",
        "pca":"PCA",
        "scgpt":"scGPT",
        "scimilarity":"SCimilarity",
        'scMulan':'scMuLan',
        'uce':"UCE",
        'scfoundation':'scFoundation',
        'scvi':'scVI',
        'linearscvi':'LDVAE',
        'cellfishing':"CellFishing.jl"
    }

    df = {"model":[],"aucprc":[]}
    df_roc = {"model":[],"aucroc":[]}

    average_precision_dict = dict()
    precision_dict = dict()
    accuracy_dict = dict()
    recall_dict = dict()
    f1_dict = dict()

    for platform in range(1,10):
        try:
            average_precision_dict[platform] = dict()
            precision_dict[platform] = dict()
            accuracy_dict[platform] = dict()
            recall_dict[platform] = dict()
            f1_dict[platform] = dict()

            y_score_all = []
            for model in models:
                seen = np.mean(np.load(os.path.join(scimilarity_path,f"{model}_{platform}_{args.cell_type}","seen_cosine_similarity.npy"))[:,0:args.number],axis=1)
                unseen = np.mean(np.load(os.path.join(scimilarity_path,f"{model}_{platform}_{args.cell_type}","unseen_cosine_similarity.npy"))[:,0:args.number],axis=1)
                seen = seen.flatten()
                unseen = unseen.flatten()
                
                if model == 'cellfishing':
                    seen = 1 / seen
                    unseen = 1 / unseen
                y_true = np.append(np.ones(len(unseen)),np.zeros(len(seen)))
                y_score = np.append(unseen,seen)
                y_score_all.append(y_score)


            np.random.seed(42)  # For reproducibility
            models_dict = dict()
            for i in range(len(models)):
                models_dict[name_dict[models[i]]] = y_score_all[i]

            for model_name, y_scores in models_dict.items():
                thres = np.percentile(y_scores, args.threshold)
                precision_dict[platform][model_name]=precision_score(y_true=y_true, y_pred=y_scores<thres)
                recall_dict[platform][model_name]= recall_score(y_true=y_true, y_pred=y_scores<thres)
                f1_dict[platform][model_name] = f1_score(y_true=y_true, y_pred=y_scores<thres)
                accuracy_dict[platform][model_name] = accuracy_score(y_true=y_true, y_pred=y_scores<thres)
                # average_precision_dict[platform][model_name] = average_precision_score(y_true, y_scores)
                # auc_roc_dict[platform][model_name] = roc_auc_score(y_true, y_scores)
        except:
            pass

    to_del = []
    for i in precision_dict.keys():
        if len(precision_dict[i]) == 0:
            to_del.append(i)

    new_precision_dict = dict()
    for i in precision_dict.keys():
        if i not in to_del:
            new_precision_dict[i] = precision_dict[i]
            

    precision_df = pd.DataFrame(new_precision_dict)
    os.makedirs("precision_" + str(args.number) + "_" + str(args.threshold) + "/",exist_ok=True)
    precision_df.to_csv("precision_" + str(args.number) + "_" + str(args.threshold) + "/" + args.cell_type + ".csv")

    to_del = []
    for i in accuracy_dict.keys():
        if len(accuracy_dict[i]) == 0:
            to_del.append(i)

    new_accuracy_dict = dict()
    for i in accuracy_dict.keys():
        if i not in to_del:
            new_accuracy_dict[i] = accuracy_dict[i]
            

    accuracy_df = pd.DataFrame(new_accuracy_dict)
    os.makedirs("accuracy_" + str(args.number) + "_" + str(args.threshold) + "/",exist_ok=True)
    accuracy_df.to_csv("accuracy_" + str(args.number) + "_" + str(args.threshold) + "/" + args.cell_type + ".csv")

    to_del = []
    for i in recall_dict.keys():
        if len(recall_dict[i]) == 0:
            to_del.append(i)

    new_recall_dict = dict()
    for i in recall_dict.keys():
        if i not in to_del:
            new_recall_dict[i] = recall_dict[i]

    to_del = []
    for i in f1_dict.keys():
        if len(f1_dict[i]) == 0:
            to_del.append(i)

    new_f1_dict = dict()
    for i in f1_dict.keys():
        if i not in to_del:
            new_f1_dict[i] = f1_dict[i]

    recall_df = pd.DataFrame(new_recall_dict)
    f1_df = pd.DataFrame(new_f1_dict)

    os.makedirs("recall_" + str(args.number) + "_" + str(args.threshold) + "/",exist_ok=True)
    recall_df.to_csv("recall_" + str(args.number) + "_" + str(args.threshold) + "/" + args.cell_type + ".csv")
    os.makedirs("f1_" + str(args.number) + "_" + str(args.threshold) + "/",exist_ok=True)
    f1_df.to_csv("f1_" + str(args.number) + "_" + str(args.threshold) + "/" + args.cell_type + ".csv")