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

### PBMC-Ding
PBMC-Ding is a multi-platform human blood dataset with 310,021 cells from adult stage, using multiple protocols (9 in total), designed for multiple splits by platform evaluation where one platform is excluded as query and others as reference.

### Pancreas
Pancreas is a multi-platform human pancreas dataset with 6,321 adult-stage cells, employing multiple protocols (4 in total), and used in multiple splits by platform settings alongside PBMC-Ding.

### Cerebellum-dev
Cerebellum-dev is a multi-species (human/mouse) cerebellum dataset with 294,363 cells from development stage, processed via 10x protocol, and evaluated under multi-species retrieval settings.

### Muscle-aging
Muscle-aging is a multi-species (human/mouse) muscle dataset with 234,230 cells from aging stage, using 10x protocol, and included in multi-species evaluations.

### Tabula
Tabula is a multi-species (human/mouse) dataset covering multiple organs with 107,071 adult-stage cells and multiple protocols, primarily used for multiple random splits evaluations in cross-species retrieval tasks repeated five times.

### Arterial-Aorta
Arterial-Aorta is a multi-tech human aorta dataset with 64,141 cells from adult/aging stage, utilizing Slide-seq/scRNA protocols, and assessed in multi-technologies settings.

### Arterial-RCA
Arterial-RCA is a multi-tech human RCA dataset with 26,461 cells from adult/aging stage, based on Slide-seq/scRNA protocols, and evaluated within multi-technologies contexts.

### Limb-Embryo
Limb-Embryo is a multi-tech human limb dataset with 82,793 cells from development stage, employing Visium/scRNA protocols, and part of multi-technologies evaluations.

### Ionocyte
Ionocyte is a novel cell discovery dataset from human lung/bronchus with 92,939 adult-stage cells, processed via 10x protocol.

### COVID
COVID is a novel cell discovery dataset from human blood with 38,686 adult-stage cells, using 10x protocol.

### AD-Atlas
AD-Atlas is a disease application dataset focused on human brain with 1,329,019 cells from aging stage, incorporating multiple protocols.

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
