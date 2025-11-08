import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



cell_types = np.array(['CD16+ monocyte', 'CD4+ T cell', 'Natural killer cell', 'Dendritic cell',
                       'Cytotoxic T cell', 'Megakaryocyte', 'CD14+ monocyte', 'Plasmacytoid dendritic cell', 'B cell'])

all_scores_by_k = []
for k in [1,2,3,4,5,10,20,30]:
    all_scores = []
    for j in cell_types:
        score = pd.read_csv("scores_" + str(k) + "/" + j + ".csv")
        
        all_scores.append(score.mean(numeric_only=True,axis=1))
    all_scores = pd.concat(all_scores,axis=1)
    all_scores.index = score['Unnamed: 0']
    all_scores.columns = cell_types
    print(all_scores.mean(axis=1))
    all_scores_by_k.append(all_scores.mean(axis=1))
all_scores_by_k = pd.concat(all_scores_by_k,axis=1)
all_scores_by_k.columns = [1,2,3,4,5,10,20,30]
all_scores_by_k.to_csv("pbmc_together.csv")
breakpoint()