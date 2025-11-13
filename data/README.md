## 🧬 Datasets

Currently included datasets:

```
├─ PBMC-Ding
├─ Pancreas
├─ Cerebellum-dev
├─ Muscle-aging
├─ Tabula
├─ Arterial-Aortia
├─ Arterial-RCA
├─ Limb-Embryo
├─ Ionocyte
└─ COVID
```
Noted that, for cross-species datasets, mouse-to-human gene mapping data are provided. The mapping is extracted, using the API provided by [BioMart](https://www.ensembl.org/info/data/biomart/index.html).

Different settings are applied to different datasets, we provide retrieval scripts based on the settings defined in our benchmark:

| **Setting**              | **Dataset(s)**                          |
|-------------------------------------|-----------------------------------------|
| Multiple Splits by Platforms, Multi-Platform | PBMC-Ding, Pancreas                    |
| Multiple Random Splits Evaluations, Multi-Species  | Tabula                                 |
| Multi-Species                       | Cerebellum-dev, Muscle-aging           |
| Multi-Technologies                  | Arterial-Aorta, Arterial-RCA, Limb-Embryo |
| Novel Cell Discovery           | Ionocyte, COVID                        |

#### Multiple Splits by Platform

One platform is excluded as the query dataset, while the others are used as the reference for retrieval evaluation.

#### Multiple Random Splits Evaluations

For each cross-species retrieval task, we designate one species as the query set and another as the reference set, then randomly sample to create splits, repeating this process five times for evaluation.

---
## Steps to download and organize data

1. All included datasets can be downloaded from the Google Drive link [here](https://drive.google.com/drive/folders/1BpVqwQc2uCDatsNCr-QTUDxt2oWbRt43?usp=sharing).

2. For each dataset, unzip the compressed files to obtain raw h5ad files and model inference embeddings.

Example (PBMC-Ding)
```bash
unzip PBMC-Ding.zip #raw h5ad files
cd ./PBMC-Ding
mkdir ./embeddings
cd ./embeddings
unzip PBMC-Ding.zip #model inference embeddings
```
