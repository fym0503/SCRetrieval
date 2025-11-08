import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

rcParams['font.family'] = 'Arial'

df = pd.read_csv("full_result_deg_overlap.csv")
df = df.set_index("Unnamed: 0")  # 设置索引


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
    'cellfishing':'CellFishing.jl'
}

new_index = []
for i in df.columns:
    new_index.append(name_dict[i])
    
df.columns = new_index
df.index = new_index
# 创建上三角掩码
df = df[list(df.columns[:-1])][1:]
mask = np.triu(np.ones_like(df, dtype=bool))  # 创建上三角掩码
for i in range(len(mask)):
    mask[i,i] = False
# 绘制热图
plt.figure(figsize=(8, 8))
ax = sns.heatmap(
    df,
    mask=mask,               # 遮掩上三角部分
    cmap="Blues",             # 使用红蓝配色
    annot=True,              # 显示数值
    fmt=".2f",               # 保留两位小数
    cbar=False,               # 禁用 colorbar
    annot_kws={"fontsize": 17}

)

# 移除刻度线
ax.tick_params(left=False, bottom=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.tight_layout()
plt.savefig("full_result_deg_overlap.pdf",transparent=True)
