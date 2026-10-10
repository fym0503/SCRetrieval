## Evaluation Scripts

Run the evaluation scripts to anaylze the retrieval performance of each model across different datasets based on different settings.

| **Setting**              | **Dataset(s)**                          |
|-------------------------------------|-----------------------------------------|
|  Multiple Splits by Platforms, Multi-Platform | PBMC-Ding, Pancreas                    |
| Multiple Random Splits Evaluations, Multi-Species  | Tabula                                 |
| Multi-Species                       | Cerebellum-dev, Muscle-aging           |
| Multi-Technologies                  | Arterial-Aorta, Arterial-RCA, Limb-Embryo |
| Novel Cell Discovery           | Ionocyte, COVID                        |

All the metrics used in the benchmark are included, measuring Retrieval Accuracy, Retrieval Diversity and Retrieval Fairness.

### Retrieval Accuracy
- **VoteAcc**: Majority vote of top-K cell types matches query
- **AvgAcc**: Proportion of correct cell types in top-K
- **MAP**: Mean Average Precision
- **NDCG**: Ranking quality with position weighting

### Retrieval Diversity
- **Entropy**: Shannon entropy of batch labels in top-K
- **α-NDCG**: NDCG with batch diversity reward (α controls redundancy)

### Retrieval Fairness
- **GiniIdx**: Gini coefficient over query-level performance
- **GiniIdx-CT**: Gini coefficient over cell-type-level performance


## Corrected retrieval metrics (K = 50, alpha = 0.5)

The original script names, input ranked CSVs, split aggregation, and result
filenames are retained. Run scripts from this directory, as before.
The metrics themselves only require NumPy and pandas; no foundation-model
environment or embedding inference is needed.

### Complete reference metadata is required

Each result directory must additionally contain `reference_obs.csv`: one row
per cell in the **complete eligible reference pool for that exact split,
direction and tissue**. Keep duplicate labels: different cells with the same
labels are separate candidates. Exclude query cells if the original retrieval
excluded them. Do not use the returned Top-50 cells or the union of retrieved
cells as the reference pool.

`retrieval/main.py` now exports this file alongside the existing `index.csv`,
`distances.npy`, and ranked annotation CSVs. Retrieval and those existing
exports are unchanged. Include both the cell-type and batch columns in
`--obs`. As in the existing ranked exports, `donor_id` is named `batch`.

| Existing driver | Reference cell-type column | Reference batch column |
| --- | --- | --- |
| PBMC-Ding | CellType | Method |
| Pancreas | celltype | tech |
| Multi-Species | cell_type | assay |
| Multi-Tech | cell_type | batch |
| Tabula | cell_type | assay |

For already saved retrievals, export metadata from the **original eligible
reference IDs**; there is no need to rerun retrieval or infer embeddings.
For example, after loading the original AnnData and the actual target indices:

```python
reference_obs = adata.obs.iloc[target_indexes][["CellType", "Method"]]
reference_obs.to_csv(result_directory / "reference_obs.csv", index=False)
```

For paired datasets, use the complete reference AnnData obs, respecting any
original species/tissue filters. Never guess the reference pool from the ranked
CSV. Missing metadata or counts inconsistent with the ranked results raise
errors. Correct the original metadata/index mapping instead of inventing
reference cells. Count checks cannot detect duplicate cell IDs or an incorrect
cell-to-label mapping: retain and inspect `index.csv` for those audits.

### Definitions

Let r_ik = 1 when the cell type at rank k matches query i, and 0 otherwise.
All scores use the first K = 50 ranks (or all returned ranks if fewer exist).

* NDCG_i@K = [sum_(k=1)^K r_ik / log2(k+1)] /
  [sum_(k=1)^min(K,R_i) 1 / log2(k+1)], where R_i is the number of
  matching cells in the complete eligible reference pool. IDCG is not
  constructed by sorting the observed retrieved relevance labels.
* Normalized entropy_i@K = -sum_b p_ib log2(p_ib) / log2(B), where
  B is the number of distinct batches in the complete eligible reference
  pool, and p_ib is the fraction of returned Top-K cells in batch b.
  Set entropy to 0 when B <= 1. This metric measures batch diversity
  among **all** retrieved cells; it does not measure biological correctness.
* alpha-DCG_i@K = sum_(k=1)^K
  r_ik (1-alpha)^(n_i,b_ik(k-1)) / log2(k+1), where n counts only
  **correctly matched** cells in the same batch at earlier ranks.
  Incorrect matches occupy ranks but do not consume novelty.
  alpha-NDCG divides this by the ideal alpha-DCG constructed from all
  matching reference cells, respecting the available count in each batch.
  The ideal greedily chooses an available batch with the highest remaining
  novelty gain. With one batch per cell and binary relevance, this produces
  the optimal ideal. alpha = 0.5; the supported range is 0 <= alpha < 1.
  At alpha = 0 it reduces to binary NDCG.

These are project implementations, not an official third-party metric library.
The corrected numerical core is implemented in this same `metrics.py` module;
there is no second metrics implementation to keep synchronized.

If R_i = 0, NDCG and alpha-NDCG are undefined (NaN), excluded from their
means and Gini calculations. The two added rows `nDCG_valid_fraction` and
`α-NDCG_valid_fraction` report the fraction of evaluable queries.
All-undefined means remain NaN. For consistency with the reproduction code,
Gini returns 0 when no finite scores remain; this convention must be interpreted
together with the valid fraction, not as evidence of good performance.
Existing scripts retain equal weighting across splits when averaging metrics.

### API and result compatibility

The existing dictionary key **Average Entropy** is retained for plotting
compatibility, but now contains **normalized** entropy. NDCG and alpha-NDCG
also keep their existing keys. Their six associated Gini values are recalculated
from the revised per-query scores. Do not mix older and revised CSVs under the
same interpretation. The two valid-fraction rows are metadata: exclude them
from composite performance scores.

```python
from metrics import (
    compute_cell_type_matching_metrics, compute_batch_diversity_metrics,
    compute_balanced_metrics, load_reference_metadata,
)
reference_labels, reference_batches = load_reference_metadata(
    "reference_obs.csv", "cell_type", "batch"
)
matching = compute_cell_type_matching_metrics(
    queries, retrieved_labels, k=50, reference_labels=reference_labels
)
diversity = compute_batch_diversity_metrics(
    queries, retrieved_labels, retrieved_batches, alpha=0.5, k=50,
    reference_labels=reference_labels, reference_batches=reference_batches
)
fairness = compute_balanced_metrics(
    queries, retrieved_labels, k=50, reference_labels=reference_labels
)
```

Reference arrays are mandatory for corrected metrics. There is no fallback to
the old formulas. New reference and cutoff parameters are keyword-only in the
diversity API; alpha retains its old positional slot but is validated.
The benchmark drivers now explicitly pass `alpha=0.5, k=k`; previously several
passed the cutoff as alpha.

Vote accuracy, average accuracy, the existing MAP convention, MRR, and their
eight Gini values retain their original implementations. Existing lower-level
default cutoffs for matching functions are retained for API compatibility;
the benchmark scripts and examples explicitly use 50.

### Tests

From the repository root:

```bash
python -m unittest discover -s tests -p test_retrieval_metrics.py -v
```

Tests include hand-calculated values, reference-based denominators, batch
novelty, exhaustive small-pool alpha-IDCG verification, K/alpha validation,
absent reference types, driver arguments, and the reference metadata export.
