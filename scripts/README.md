## Retrieval

Run the retrieval experiments for datasets and models included in the bencmark.

Please generate models' embeddings before running the retrieval scripts. You can also use the raw data and embeddings provided by us, following the setup instructions in **data** directory.

To obtain the retrieval results for all models in PBMC-Ding dataset, run:

```bash
bash PBMC-Ding_retrieval.sh
```
## Evaluation

To obtain the metrics, Retrieval Accuracy, Retrieval Diversity and Retrieval Fairness, run:

```bash
bash run_metric.sh
```
