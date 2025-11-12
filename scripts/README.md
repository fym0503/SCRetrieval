## Retrieval

Run the retrieval experiments for datasets and models included in the bencmark.

Please generate models' embeddings before running the retrieval scripts. You can also use the raw data and embeddings provided by us, following the setup instructions in **data** directory.

To obtain the retrieval results for all models in PBMC-Ding dataset, run:

```bash
bash PBMC-Ding_retrieval.sh
```

Different settings are applied to different datasets, we provide retrieval scripts based on the settings defined in our benchmark:

| **Setting**              | **Dataset(s)**                          |
|-------------------------------------|-----------------------------------------|
|  Multiple Splits by Platforms, Multi-Platform | PBMC-Ding, Pancreas                    |
| Multiple Random Splits Evaluations, Multi-Species  | Tabula                                 |
| Multi-Species                       | Cerebellum-dev, Muscle-aging           |
| Multi-Technologies                  | Arterial-Aorta, Arterial-RCA, Limb-Embryo |
| Novel Cell Discovery           | Ionocyte, COVID                        |

## Evaluation

To obtain the metrics, Retrieval Accuracy, Retrieval Diversity and Retrieval Fairness, run:

```bash
bash run_metric.sh
```
