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
