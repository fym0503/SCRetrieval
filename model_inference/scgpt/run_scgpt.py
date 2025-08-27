import sys
from pathlib import Path
import scipy 
import numpy as np
import pandas as pd
import scanpy as sc
import argparse
import os
import time
sys.path.insert(0, "../")
import scgpt as scg
from scipy.sparse import csr_matrix

parser = argparse.ArgumentParser(description='Process single-cell data.')
parser.add_argument('--input_adata', type=str, required=True, help='Path to input .h5ad file')
parser.add_argument('--output_path', type=str, required=True, help='Path to output .npy file')
parser.add_argument('--model_path', type=str, required=True, help='Path to scgpt model')
args = parser.parse_args()

model_dir=Path(args.model_path)
adata=sc.read_h5ad(args.input_adata)

cell_type_key = "CellType"
gene_col = "feature_name"
batch_key = "Method"
N_HVG = 3000
org_adata = adata.copy()
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor='seurat_v3')
adata = adata[:, adata.var['highly_variable']]

zero_rows = np.all(adata.X.toarray() == 0, axis=1)
print(np.sum(zero_rows))
# Convert sparse matrix to dense array
dense_X = adata.X.toarray()

# Add a small number (e.g., 1e-10) to the first column of zero rows
dense_X[zero_rows, 0] += 1e-10

# Convert back to sparse matrix
adata.X = csr_matrix(dense_X)


start_cpu_time = time.process_time()
ref_embed_adata = scg.tasks.embed_data( 
    adata,
    model_dir,
    gene_col=gene_col,
    #obs_to_save=cell_type_key, 
    # optional arg, only for saving metainfo
    batch_size=10,
    #return_new_adata=True,
    # device='cpu',
)
execution_time = time.process_time() - start_cpu_time

print(f"Embedding extraction time in seconds: {execution_time:.6f} seconds")
#ref_embed_adata.write(args.output_path)
np.save(args.output_path,ref_embed_adata.obsm['X_scGPT'])

# modified embed_data
'''
def embed_data(
    adata_or_file: Union[AnnData, PathLike],
    model_dir: PathLike,
    gene_col: str = "feature_name",
    max_length=1200,
    batch_size=64,
    obs_to_save: Optional[list] = None,
    device: Union[str, torch.device] = "cuda",
    use_fast_transformer: bool = True,
    return_new_adata: bool = False,
) -> AnnData:
    """
    Preprocess anndata and embed the data using the model.

    Args:
        adata_or_file (Union[AnnData, PathLike]): The AnnData object or the path to the
            AnnData object.
        model_dir (PathLike): The path to the model directory.
        gene_col (str): The column in adata.var that contains the gene names.
        max_length (int): The maximum length of the input sequence. Defaults to 1200.
        batch_size (int): The batch size for inference. Defaults to 64.
        obs_to_save (Optional[list]): The list of obs columns to save in the output adata.
            Useful for retaining meta data to output. Defaults to None.
        device (Union[str, torch.device]): The device to use. Defaults to "cuda".
        use_fast_transformer (bool): Whether to use flash-attn. Defaults to True.
        return_new_adata (bool): Whether to return a new AnnData object. If False, will
            add the cell embeddings to a new :attr:`adata.obsm` with key "X_scGPT".

    Returns:
        AnnData: The AnnData object with the cell embeddings.
    """
    if isinstance(adata_or_file, AnnData):
        adata = adata_or_file
    else:
        adata = sc.read_h5ad(adata_or_file)

    if isinstance(obs_to_save, str):
        assert obs_to_save in adata.obs, f"obs_to_save {obs_to_save} not in adata.obs"
        obs_to_save = [obs_to_save]

    # verify gene col
    if gene_col == "index":
        adata.var["index"] = adata.var.index
    else:
        assert gene_col in adata.var

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Using CPU instead.")

    # print(device)

    # LOAD MODEL
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    # vocabulary
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    # Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.

    vocab.set_default_index(vocab["<pad>"])
    genes = adata.var[gene_col].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # all_counts = adata.layers["counts"]
    # num_of_non_zero_genes = [
    #     np.count_nonzero(all_counts[i]) for i in range(all_counts.shape[0])
    # ]
    # max_length = min(max_length, np.max(num_of_non_zero_genes) + 1)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=model_configs["pad_token"],
        pad_value=model_configs["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=use_fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=False,
    )
    
    # if device!=torch.device("cpu"):
    #     load_pretrained(model, torch.load(model_file), verbose=False)
    # else:
    load_pretrained(model, torch.load(model_file, map_location=torch.device('cpu')), verbose=False)
        
    model.to(device)
    model.eval()

    # get cell embeddings
    cell_embeddings = get_batch_cell_embeddings(
        adata,
        cell_embedding_mode="cls",
        model=model,
        vocab=vocab,
        max_length=max_length,
        batch_size=batch_size,
        model_configs=model_configs,
        gene_ids=gene_ids,
        use_batch_labels=False,
    )

    if return_new_adata:
        obs_df = adata.obs[obs_to_save] if obs_to_save is not None else None
        return sc.AnnData(X=cell_embeddings, obs=obs_df, dtype="float32")

    adata.obsm["X_scGPT"] = cell_embeddings
    return adata

'''