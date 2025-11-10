import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

df = pd.read_csv("./scimilarity_metrics_abl.csv")

breakpoint()
all_metrics = [
    'Top-K Vote Accuracy', 'Top-K Average Accuracy', 'Mean Average Precision (MAP)', 
    'Normalized Discounted Cumulative Gain (nDCG)', 'Mean Reciprocal Rank (MRR)', 
    'Top-K Vote Accuracy_gini', 'Top-K Vote Accuracy_celltype_gini', 
    'Top-K Average Accuracy_gini', 'Top-K Average Accuracy_celltype_gini', 
    'MAP_gini', 'MAP_celltype_gini', 'nDCG_gini', 'nDCG_celltype_gini', 
    'MRR_gini', 'MRR_celltype_gini'
]

n_probes = np.sort(np.unique(df['n_probe']).astype(int))
n_centroids_values = np.sort(np.unique(df['n_centroids']).astype(int))

def create_average_heatmap(metric):


    plt.figure(figsize=(10, 8))
    
    colors = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", 
              "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    cmap_name = 'custom_diverging'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    result_matrix = np.zeros((len(n_probes), len(n_centroids_values)))
    count_matrix = np.zeros((len(n_probes), len(n_centroids_values)))
    
    for i, probe in enumerate(n_probes):
        for j, centroids in enumerate(n_centroids_values):
            mask = (df['n_probe'] == probe) & (df['n_centroids'] == centroids)
            filtered_data = df.loc[mask, metric]
            
            if not filtered_data.empty:
                result_matrix[i, j] = filtered_data.mean()
                count_matrix[i, j] = len(filtered_data)
    
    non_zero_values = result_matrix[result_matrix > 0]
    if len(non_zero_values) > 0:
        vmin = non_zero_values.min()
        vmax = non_zero_values.max()
    else:
        vmin = None
        vmax = None
    
    has_missing = np.any(count_matrix == 0)
    
    title = f'Average {metric} across All Protocols'
    if has_missing:
        title += "\n(White cells indicate missing data)"
    
    ax = sns.heatmap(
        result_matrix,
        cmap=cm,
        vmin=vmin, 
        vmax=vmax, 
        annot=True,
        fmt='.3f',
        linewidths=0.5,
        xticklabels=n_centroids_values,
        yticklabels=n_probes,
        cbar_kws={'label': metric}
    )
    
    if has_missing:
        mask = count_matrix == 0
        for i in range(len(n_probes)):
            for j in range(len(n_centroids_values)):
                if mask[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='white', alpha=1))
                    ax.text(j + 0.5, i + 0.5, 'N/A', ha='center', va='center')
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('n_centroids', fontsize=12, labelpad=10)
    plt.ylabel('n_probe', fontsize=12, labelpad=10)
    
    plt.tight_layout()
    
    safe_metric_name = metric.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f'average_heatmap_{safe_metric_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    return result_matrix

for metric in all_metrics:
    print(f"Creating average heatmap for {metric}...")
    create_average_heatmap(metric)

print("All average heatmaps have been created!")