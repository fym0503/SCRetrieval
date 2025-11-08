import os
import pandas as pd
import numpy as np

# cell_types = np.array(['CD16+ monocyte', 'CD4+ T cell', 'Natural killer cell', 'Dendritic cell',
#                        'Cytotoxic T cell', 'Megakaryocyte', 'CD14+ monocyte', 'Plasmacytoid dendritic cell', 'B cell'])

cell_types = np.array(['CD16+ monocyte', 'CD4+ T cell', 'Dendritic cell',
                       'Cytotoxic T cell', 'CD14+ monocyte', 'Plasmacytoid dendritic cell'])

platform_class = {1: '10x Chromium (v2)', 2: '10x Chromium (v2) A', 3: '10x Chromium (v2) B', 4: '10x Chromium (v3)', 5: 'CEL-Seq2', 6: 'Drop-seq', 7: 'Seq-Well', 8: 'Smart-seq2', 9: 'inDrops'}


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
df1_sorted.to_csv("all_data.csv")
