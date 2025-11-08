import statsmodels.api as sa
import scikit_posthocs as sp
import pandas as pd
import numpy as np
import os
import scikit_posthocs as sp
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
plt.rcParams['font.family'] = 'Arial'

data_all = pd.read_csv("full_result.csv")
del data_all['Unnamed: 0.1']
data_all = data_all.set_index("Unnamed: 0")
del data_all['Row_Sum']

data_all = data_all.T
dict_data = dict()
for key in data_all.keys():
    dict_data[key] = data_all[key].values

data = (
    pd.DataFrame(dict_data)
    .rename_axis('fold')
    .melt(
        var_name='datasets',
        value_name='score',
        ignore_index=False,
    )
    .reset_index()
)

avg_rank = data.groupby('fold').score.rank(pct=True).groupby(data.datasets).mean()

import scipy.stats as ss
import matplotlib.pyplot as plt
ss.friedmanchisquare(*dict_data.values())
test_results = sp.posthoc_conover_friedman(
    data,
    melted=True,
    block_col='fold',
    group_col='datasets',
    y_col='score',
)
plt.figure(figsize=(7.5, 4), dpi=900)
sp.critical_difference_diagram(avg_rank, test_results,label_fmt_left="{label}",label_fmt_right ="{label}" )
plt.savefig('cdgraph.pdf')