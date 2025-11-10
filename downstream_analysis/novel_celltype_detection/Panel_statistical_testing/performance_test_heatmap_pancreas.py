import statsmodels.api as sa
import scikit_posthocs as sp
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import scikit_posthocs as sp
import scipy
from scipy.stats import wilcoxon
import seaborn as sns

plt.rcParams["font.family"] = "Arial"

files = os.listdir("novel_cell_type/Panel_quantitative_benchmark/Pancreas/scores")

df = []
for f in files:
    df_i = pd.read_csv("novel_cell_type/Panel_quantitative_benchmark/Pancreas/scores/" + f)
    df_i = df_i.set_index("Unnamed: 0")
    df.append(df_i)

df = pd.concat(df,axis=1)
row_means = df.mean(axis=1)
df = df.reindex(row_means.sort_values(ascending=False).index)

methods = list(df.index)
df = df.T



p_value_matrix = pd.DataFrame(np.ones((len(methods), len(methods))), columns=methods, index=methods)

symbol_matrix = pd.DataFrame("", columns=methods, index=methods)

for i, method1 in enumerate(methods):
    for j, method2 in enumerate(methods):
        if i < j:  #
            
            stat, p = wilcoxon(df[method1], df[method2], alternative='two-sided',correction=True)
            p_value_matrix.loc[method1, method2] = p
            p_value_matrix.loc[method2, method1] = p

            # 
            diff = np.mean(df[method1]) - np.mean(df[method2])
            direction = "↑" if diff > 0 else "↓"

            #
            if p < 0.01:
                symbol_matrix.loc[method1, method2] = f"**{direction}"
                symbol_matrix.loc[method2, method1] = f"**{'↑' if direction == '↓' else '↓'}"
            elif p < 0.05:
                symbol_matrix.loc[method1, method2] = f"*{direction}"
                symbol_matrix.loc[method2, method1] = f"*{'↑' if direction == '↓' else '↓'}"
            else:
                symbol_matrix.loc[method1, method2] = f"-"
                symbol_matrix.loc[method2, method1] = f"-"
#
heatmap_data = p_value_matrix.copy()
heatmap_data[heatmap_data > 0.05] = 0.05 

heatmap_data = heatmap_data[list(heatmap_data.columns[1:])][:-1]
mask = np.tril(np.ones_like(heatmap_data, dtype=bool))  # 创建上三角掩码
for i in range(len(mask)):
    mask[i,i] = False
    
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    heatmap_data,
    mask = mask,
    annot=symbol_matrix[list(symbol_matrix.columns[1:])][:-1], 
    annot_kws={"size": 25},  # 
    fmt="",  # 
    cmap="Blues_r",  #  p
    cbar=False, square=True

)



ax.xaxis.set_label_position('top')           # 将 X 轴标签移动到顶部
ax.xaxis.tick_top()                          # 设置 X 轴刻度在顶部
plt.xticks(fontsize=20, rotation=90)         # 设置 X 轴标签旋转 90 度

ax.yaxis.set_label_position('right')         # 将 Y 轴标签移动到右侧
ax.yaxis.tick_right()                        # 设置 Y 轴刻度在右侧
plt.yticks(fontsize=20, rotation=0)          # 设置 Y 轴标签旋转 90 度
ax.tick_params(right=False, top=False)

plt.xticks(fontsize=20,rotation=90)  #  
plt.yticks(fontsize=20,rotation=0)  # 
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig("heatmap_Pancreas.pdf",dpi=900,transparent=True)
