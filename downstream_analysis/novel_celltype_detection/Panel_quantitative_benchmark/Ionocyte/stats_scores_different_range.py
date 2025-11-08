import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# cell_types = np.array(['acinar','activated_stellate','alpha','beta','delta','ductal','endothelial',
#                        'epsilon','gamma','macrophage','mast','quiescent_stellate', 'schwann'])

all_scores_by_k = []
for k in [1,2,3,4,5,10,20,30]:
    
    score = pd.read_csv("all_data_" + str(k) + ".csv")
        
    all_scores_by_k.append(score['mean'])
    
    # all_scores_by_k = pd.concat(all_scores_by_k,axis=1)
    # all_scores_by_k.index = score['Unnamed: 0']
    # all_scores_by_k.columns = cell_types
    # print(all_scores.mean(axis=1))
    # all_scores_by_k.append(all_scores.mean(axis=1))
all_scores_by_k = pd.concat(all_scores_by_k,axis=1)
all_scores_by_k.index = score['Unnamed: 0']
all_scores_by_k.columns = [1,2,3,4,5,10,20,30]
all_scores_by_k.to_csv("ionocyte_together.csv")
breakpoint()