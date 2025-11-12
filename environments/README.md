## 1. Clone the Repository

Clone the SCRetrieval repository to your local machine using the following command:

```bash
git clone https://github.com/fym0503/SCRetrieval.git
```

## 2. Environment Setup

Each method in SCRetrieval requires specific Python or Julia versions and packages. You can set up the required environments by running the provided installation scripts (`[env]_installation.sh`) for each method.

The table below summarizes the virtual environments, required Python/Julia versions, and corresponding methods:

| Installation Script       | Python/Julia Version | Methods Supported                              |
|---------------------------|----------------------|------------------------------------------------|
| `env1_installation.sh`    | Python 3.8           | UCE, SCimilarity                               |
| `env2_installation.sh`    | Python 3.11          | Geneformer, scFoundation, scVI, PCA, LDVAE     |
| `env3_installation.sh`    | Python 3.10          | scGPT                                          |
| `env4_installation.sh`    | Python 3.10          | scMulan                                        |
| `env5_installation.sh`    | Python 3.9           | CellPLM                                        |
| `env6_installation.sh`    | Python 3.9           | CellBlast                                      |
| `env7_installation.sh`    | Julia 1.6.7          | CellFishing.jl                                 |

### Setup Instructions

1. Navigate to the cloned repository directory:
   ```bash
   cd SCRetrieval/environments
   ```

2. Run the appropriate installation script for the method you want to use. For example:
   ```bash
   bash env1_installation.sh
   ```

   This will install the required dependencies for the specified method.

3. Repeat for other methods as needed, ensuring the correct virtual environment is activated for each.
