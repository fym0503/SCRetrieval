import os
import pandas as pd
import numpy as np


cell_types = np.array(['DC_c3-LAMP3','Macro_c3-EREG', 'Macro_c4-DNAJB1',
       'Neu_c4-RSAD2', 'Neu_c6-FGF23'])

values = []
for i in cell_types:
    df = pd.read_csv("scores/" + i + ".csv")
    values.append(list(df['0']))

values = np.array(values)
df1 = pd.DataFrame(values).T
df1.index = list(df['Unnamed: 0'])
row_means = df1.mean(axis=1)

df1_sorted = df1.reindex(row_means.sort_values(ascending=False).index)
df1_sorted['mean'] = list(row_means.sort_values(ascending=False))
df1_sorted.to_csv("all_data.csv")