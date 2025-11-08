import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

# 设置字体为 Arial
rcParams['font.family'] = 'Arial'
name_dict = {
    "cellblast": "CellBlast",
    "cellplm": "CellPLM",
    "geneformer": "Geneformer",
    "pca": "PCA",
    "scgpt": "scGPT",
    "scimilarity": "SCimilarity",
    'scMulan': 'scMuLan',
    'uce': "UCE",
    'scfoundation': 'scFoundation',
    'scvi': 'scVI',
    'linearscvi': 'LDVAE',
    'cellfishing': 'CellFishing.jl'
}
# 读取数据
df = pd.read_csv("full_result_marker_overlap.csv")
df = df.set_index("Unnamed: 0")  # 设置索引
new_index = []
for i in df.columns:
    new_index.append(name_dict[i])

df.columns = new_index
df.index = new_index
df = df[list(df.columns[1:])][:-1]
# 修改列和索引名称




# 创建下三角掩码
mask = np.tril(np.ones_like(df, dtype=bool))  # 创建下三角掩码
for i in range(len(mask)):
    mask[i,i] = False
# breakpoint()
# 绘制热图
plt.figure(figsize=(8, 8))
ax = sns.heatmap(
    df,
    mask=mask,               # 遮掩上三角部分
    cmap="Blues",            # 使用蓝色配色
    annot=True,              # 显示数值
    fmt=".2f",               # 保留两位小数
    cbar=False,              # 禁用 colorbar
    annot_kws={"fontsize": 17}
)

# 移除所有刻度线
# 

# 设置顶部和右侧的标签
ax.xaxis.set_label_position('top')           # 将 X 轴标签移动到顶部
ax.xaxis.tick_top()                          # 设置 X 轴刻度在顶部
plt.xticks(fontsize=20, rotation=90)         # 设置 X 轴标签旋转 90 度

ax.yaxis.set_label_position('right')         # 将 Y 轴标签移动到右侧
ax.yaxis.tick_right()                        # 设置 Y 轴刻度在右侧
plt.yticks(fontsize=20, rotation=0)          # 设置 Y 轴标签旋转 90 度
ax.tick_params(right=False, top=False)
# 调整布局并保存图表
plt.tight_layout()
plt.savefig("full_result_marker_overlap.pdf",transparent=True)
