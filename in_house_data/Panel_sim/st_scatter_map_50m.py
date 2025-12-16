import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc

adata = sc.read("../shared_data/X5704ADHC.h5ad")
distances = np.load("../shared_data/retrieval_output_50m.npy").mean(axis=1)

obs_df = adata.obs.copy().reset_index()
obs_df['distance'] = distances

p5 = np.percentile(distances, 2.5)
p95 = np.percentile(distances, 97.5)

filter_mask = (obs_df['distance'] >= p5) & (obs_df['distance'] <= p95)
obs_df_filtered = obs_df[filter_mask]

print(f"Original points: {len(obs_df)}, Filtered points: {len(obs_df_filtered)}")
print(f"Distance range (5%-95%): {p5:.3f} - {p95:.3f}")

fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(
    np.array(obs_df_filtered['center_x']),
    np.array(obs_df_filtered['center_y']),
    c=obs_df_filtered['distance'],
    cmap='RdYlBu_r',
    s=2,
    rasterized=True
)

ax.set_aspect('equal')
ax.axis('off')

cb = plt.colorbar(scatter, ax=ax)
cb.ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig('distance_heatmap.pdf', bbox_inches='tight', pad_inches=0)
plt.close()

print("Saved: distance_heatmap.pdf")