![figure](logo/scretrieval.png)

## Benchmarking Traditional Methods and Foundation Models for Single-cell Transcriptomics Data Search and Retrieval
## 📁 Directory Structure

```
.
├── data                    # Data files 
├── downstream_analysis     # Scripts for downstream tasks analysis
├── environments            # Reproducible Python or Julia environments  
├── logo                    
├── model_inference         # Models and their inference scripts included in the benchmark
├── quantitative_metrics    # Scripts for evaluating models' retrieval performance
├── retrieval               # Main scripts for retrieval
├── scripts                 # Bash scripts for running retrieval experiments and evaluations
├── LICENSE  
└── README.md  
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/fym0503/SCRetrieval.git
cd SCRetrieval
```

### 2. Set Up for Each Method

Python and Julia versions, along with required packages for each method, are summarized in the **`environments/`** directory.  
Use the provided installation scripts to set up corresponding environments.

| Installation Script       | Python/Julia Version | Methods Supported                              |
|---------------------------|----------------------|------------------------------------------------|
| `env1_installation.sh`    | Python 3.8           | UCE, SCimilarity                               |
| `env2_installation.sh`    | Python 3.11          | Geneformer, scFoundation, scVI, PCA, LDVAE     |
| `env3_installation.sh`    | Python 3.10          | scGPT                                          |
| `env4_installation.sh`    | Python 3.10          | scMulan                                        |
| `env5_installation.sh`    | Python 3.9           | CellPLM                                        |
| `env6_installation.sh`    | Python 3.9           | CellBlast                                      |
| `env7_installation.sh`    | Julia 1.6.7          | CellFishing.jl                                 |

To install the required environment for a specific setup, run:

```bash
bash environments/env1_installation.sh
```

---

## 🚀 Quick Start: 3-Step Pipeline

Follow this **end-to-end pipeline** to run model inference, perform cell retrieval, and evaluate performance using the **PBMC-Ding** dataset as an example.

---

### **Step 1: Model Inference**  
Generate cell embeddings from raw single-cell data using pre-trained models.

#### Example: Geneformer
```bash
cd model_inference/geneformer
# Ensure paths are set: model weights, input .h5ad, output embedding path
bash run_model.sh
```

#### Example: cellfishing.jl (Julia)
> Requires **Julia 1.6.7** (download the zip installer if needed)
```bash
cd model_inference/cellfishing
julia run_cellfishing.jl
```

> Output: `.npy` files with cell embeddings saved in specified directories.

---

### **Step 2: Cell Retrieval**  
Query cells using the generated embeddings and reference annotations.

#### Example: PBMC-Ding Retrieval
```bash
cd scripts
# Update embedding and .h5ad paths if needed
bash PBMC-Ding_retrieval.sh
```

> Output: Retrieval results (e.g., Indexes and similarity scores of retrieved single cells) for query cells.

---

### **Step 3: Performance Evaluation**  
Compute retrieval metrics (e.g., Retrieval Accuracy, Retrieval Diversity and Retrieval Fairness).

#### Example: Evaluate PBMC-Ding
```bash
cd scripts
bash run_metric.sh
```

> Output: Summary of retrieval performance across cell types and conditions.

---

**All datasets & precomputed embeddings** are available in the `data/` directory — follow instructions there to download.

---

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
Here is the list of different datasets' settings:

| **Setting**              | **Dataset(s)**                          |
|-------------------------------------|-----------------------------------------|
| Multi-Platform | PBMC-Ding, Pancreas                    |
| Multi-Species  | Tabula, Cerebellum-dev, Muscle-aging                                 |
| Multi-Technologies                  | Arterial-Aorta, Arterial-RCA, Limb-Embryo |
| Novel Cell Discovery           | Ionocyte, COVID                        |

Noted that, for cross-species datasets, mouse-to-human gene mapping data are provided.

---

## 🧠 Algorithms

> Current included algorithms:

### ✳️ Traditional Methods (TMs)

- [CellFishing.jl](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1639-x)
- PCA
- [scVI](https://www.nature.com/articles/s41592-018-0229-2)
- [CellBlast](https://www.nature.com/articles/s41467-020-17281-7)
- [LDVAE](https://www.nature.com/articles/s41592-018-0229-2)

### 🧩 Foundation Models (FMs)

- [Geneformer](https://www.nature.com/articles/s41586-023-06139-9)
- [scFoundation](https://www.nature.com/articles/s41592-024-02305-7)  
- [scGPT](https://www.nature.com/articles/s41592-024-02201-0)
- [UCE](https://www.biorxiv.org/content/10.1101/2023.11.28.568918v1)
- [SCimilarity](https://www.nature.com/articles/s41586-024-08411-y)
- [scMuLan](https://www.biorxiv.org/content/10.1101/2024.01.25.577152v1)
- [CellPLM](https://www.biorxiv.org/content/10.1101/2023.10.03.560734v1)


---

## 📊 Downstream Evaluation

For downstream evaluation details, refer to `downstream_analysis/README.md`.

**Currently included downstream tasks:**

- Biomarkers Discovery
- Novel Cell State Identification

---

## 📜 License
This project is licensed under the terms of the LICENSE file included in the repository.

---

## 👩‍💻 Authors & Acknowledgements

For issues or questions, please open an issue on the [GitHub repository](https://github.com/fym0503/SCRetrieval.git).

---

© 2025 SCRetrieval. All rights reserved.
