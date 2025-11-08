import os
import pandas as pd
import numpy as np

cell_types = np.array(['acinar','activated_stellate','alpha','beta','delta','ductal','endothelial',
                       'epsilon','gamma','macrophage','mast','quiescent_stellate', 'schwann'])

values = []
for i in cell_types:
    df = pd.read_csv("scores/" + i + ".csv")
    df["row_sum"] = df.iloc[:, 1:].mean(axis=1)

    values.append(list(df['row_sum']))

values = np.array(values)
df1 = pd.DataFrame(values).T
df1.index = list(df['Unnamed: 0'])
row_means = df1.mean(axis=1)

df1_sorted = df1.reindex(row_means.sort_values(ascending=False).index)
df1_sorted['mean'] = list(row_means.sort_values(ascending=False))
df1_sorted.columns = list(cell_types) + ['Average']
df1_sorted.to_csv("all_data.csv")
