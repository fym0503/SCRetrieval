import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cell_types = np.array(['DC_c3-LAMP3','Macro_c3-EREG', 'Macro_c4-DNAJB1',
       'Neu_c4-RSAD2', 'Neu_c6-FGF23'])
all_scores_by_k = []
for k in [1,2,3,4,5,10,20,30]:
    all_scores = []
    for j in cell_types:
        score = pd.read_csv("scores_" + str(k) + "/" + j + ".csv")
        all_scores.append(score['0'])
    all_scores = pd.concat(all_scores,axis=1)
    all_scores.index = score['Unnamed: 0']
    all_scores.columns = cell_types
    print(all_scores.mean(axis=1))
    all_scores_by_k.append(all_scores.mean(axis=1))
all_scores_by_k = pd.concat(all_scores_by_k,axis=1)
all_scores_by_k.columns = [1,2,3,4,5,10,20,30]
all_scores_by_k.to_csv("covid_together.csv")
breakpoint()