import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd

import torch
from anndata import AnnData
import scanpy as sc
import scvi
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import argparse

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, eval_scib_metrics, load_pretrained

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args() -> argparse.Namespace:
    r"""
    Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data", dest="input_data", type=str, required=True,
        help="path of input anndata"
    )

    parser.add_argument(
        "--pre_model", dest="pre_model", type=str, required=True,
        help="path of scgpt's pretrained model"
    )
    
    parser.add_argument(
        "--save_dir", dest="save_dir", type=str, required=True,
        help="path of output"
    )

    parser.add_argument(
        "--ecs_thres", dest="ecs_thres", type=float, default=0.8,
        help="ECS threshold"
    )

    parser.add_argument(
        "--dab_weight", dest="dab_weight", type=float, default=1,
        help="DAB weight"
    )
    
    parser.add_argument(
        "--DSBN", dest="DSBN", type=str2bool,  default=True,
        help="DSBN setting"
    )

    return parser.parse_args()

def main(args: argparse.Namespace) -> None:
    sc.set_figure_params(figsize=(4, 4))
    os.environ["KMP_WARNINGS"] = "off"
    warnings.filterwarnings('ignore')

    print("[1/6] Setting Config...")
    hyperparameter_defaults = dict(
        seed=42,
        dataset_name="Pancrease", # Dataset name
        do_train=True, # Flag to indicate whether to do update model parameters during training
        load_model=args.pre_model, # Path to pre-trained model
        GEPC=True,  # Gene expression modelling for cell objective
        ecs_thres=args.ecs_thres,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=args.dab_weight, # DAR objective weight for batch correction
        mask_ratio=0.4, # Default mask ratio
        epochs=15, # Default number of epochs for fine-tuning
        n_bins=51, # Default number of bins for value binning in data pre-processing
        lr=1e-4, # Default learning rate for fine-tuning
        batch_size=64, # Default batch size for fine-tuning
        layer_size=128,
        nlayers=4,
        nhead=4, # if load model, batch_size, layer_size, nlayers, nhead will be ignored
        dropout=0.2, # Default dropout rate during model fine-tuning
        schedule_ratio=0.9,  # Default rate for learning rate decay
        save_eval_interval=5, # Default model evaluation interval
        log_interval=100, # Default log interval
        fast_transformer=True, # Default setting
        pre_norm=False, # Default setting
        amp=True,  # # Default setting: Automatic Mixed Precision
    )

    # from types import SimpleNamespace

    # config = SimpleNamespace(**config)
    run = wandb.init(
        config=hyperparameter_defaults,
        project="scGPT",
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
        mode="offline"
    )
    config = wandb.config
    print(config)
    set_seed(config.seed)

    print("[2/6] Reading data...")
    # settings for input and preprocessing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = config.mask_ratio
    mask_value = -1
    pad_value = -2
    n_input_bins = config.n_bins

    n_hvg = 1200  # number of highly variable genes
    max_seq_len = n_hvg + 1
    per_seq_batch_sample = True
    DSBN = args.DSBN  # Domain-spec batchnorm
    explicit_zero_prob = True  # whether explicit bernoulli for zeros

    dataset_name = config.dataset_name
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")

    adata = sc.read(args.input_data)
    ori_batch_col = "tech" 
    data_is_raw = True

    adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
    batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
    adata.obs["batch_id"] = batch_id_labels
    adata.var["gene_name"] = adata.var.index.tolist()

    print("[3/6] Preprocessing...")
    if config.load_model is not None:
        model_dir = Path(config.load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        
        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will be overriden by the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
    else:
        embsize = config.layer_size 
        nhead = config.nhead
        nlayers = config.nlayers  
        d_hid = config.layer_size

    adata.layers['X'] = adata.X.copy()
    # set up the preprocessor, use the args to config the workflow
    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=3,  # step 1
        filter_cell_by_counts=False,  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key="str_batch" if dataset_name != "heart_cell" else None)

    print('Seperate train and test set')
    adata_train = adata[adata.obs['tech']!='smartseq2'].copy()
    adata_test = adata[adata.obs['tech']=='smartseq2'].copy()

    if per_seq_batch_sample:
        # sort the adata by batch_id in advance
        adata_sorted = adata_train[adata_train.obs["batch_id"].argsort()].copy()
        adata_sorted_test = adata_test[adata_test.obs["batch_id"].argsort()].copy()

    input_layer_key = "X_binned"
    all_counts = (
        adata_train.layers[input_layer_key].A
        if issparse(adata_train.layers[input_layer_key])
        else adata_train.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    celltypes_labels = adata_train.obs["celltype"].tolist()  # make sure count from 0
    all_celltypes_labels = adata.obs["celltype"].tolist()
    num_types = len(set(all_celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_train.obs["batch_id"].tolist()
    all_batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(all_batch_ids))
    batch_ids = np.array(batch_ids)

    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
    )

    print("[4/6] Loading model and tokenizing...")
    if config.load_model is None:
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )

    def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
        masked_values_train = random_mask_value(
            tokenized_train["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        masked_values_valid = random_mask_value(
            tokenized_valid["values"],
            mask_ratio=mask_ratio,
            mask_value=mask_value,
            pad_value=pad_value,
        )
        print(
            f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
            f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
        )

        input_gene_ids_train, input_gene_ids_valid = (
            tokenized_train["genes"],
            tokenized_valid["genes"],
        )
        input_values_train, input_values_valid = masked_values_train, masked_values_valid
        target_values_train, target_values_valid = (
            tokenized_train["values"],
            tokenized_valid["values"],
        )

        tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
        tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

        if sort_seq_batch:
            train_sort_ids = np.argsort(train_batch_labels)
            input_gene_ids_train = input_gene_ids_train[train_sort_ids]
            input_values_train = input_values_train[train_sort_ids]
            target_values_train = target_values_train[train_sort_ids]
            tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]

            valid_sort_ids = np.argsort(valid_batch_labels)
            input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
            input_values_valid = input_values_valid[valid_sort_ids]
            target_values_valid = target_values_valid[valid_sort_ids]
            tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]

        train_data_pt = {
            "gene_ids": input_gene_ids_train,
            "values": input_values_train,
            "target_values": target_values_train,
            "batch_labels": tensor_batch_labels_train,
        }
        valid_data_pt = {
            "gene_ids": input_gene_ids_valid,
            "values": input_values_valid,
            "target_values": target_values_valid,
            "batch_labels": tensor_batch_labels_valid,
        }

        return train_data_pt, valid_data_pt


    # dataset
    class SeqDataset(Dataset):
        def __init__(self, data: Dict[str, torch.Tensor]):
            self.data = data

        def __len__(self):
            return self.data["gene_ids"].shape[0]

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}


    # data_loader
    def prepare_dataloader(
        data_pt: Dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = False,
        intra_domain_shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        dataset = SeqDataset(data_pt)

        if per_seq_batch_sample:
            # find the indices of samples in each seq batch
            subsets = []
            batch_labels_array = data_pt["batch_labels"].numpy()
            for batch_label in np.unique(batch_labels_array):
                batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
                subsets.append(batch_indices)
            data_loader = DataLoader(
                dataset=dataset,
                batch_sampler=SubsetsBatchSampler(
                    subsets,
                    batch_size,
                    intra_subset_shuffle=intra_domain_shuffle,
                    inter_subset_shuffle=shuffle,
                    drop_last=drop_last,
                ),
                num_workers=num_workers,
                pin_memory=True,
            )
            return data_loader

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=config.GEPC,
        do_dab=True,
        use_batch_labels=True,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=DSBN,
        n_input_bins=n_input_bins,
        ecs_threshold=config.ecs_thres,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=config.fast_transformer,
        pre_norm=config.pre_norm,
    )
    if config.load_model is not None:
        load_pretrained(model, torch.load(model_file), verbose=False)

    model.to(device)
    wandb.watch(model)

    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    print("[5/6] Finetuning model...")
    def train(model: nn.Module, loader: DataLoader) -> None:
        """
        Train the model for one epoch.
        """
        model.train()
        total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
        total_error = 0.0
        total_dab, total_ecs = 0.0,0.0
        log_interval = config.log_interval
        start_time = time.time()

        num_batches = len(loader)
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    # batch_labels=batch_labels if DSBN else None,
                    batch_labels = batch_labels,
                    MVC=config.GEPC,
                    ECS=config.ecs_thres > 0,
                )

                masked_positions = input_values.eq(mask_value)  # the postions to predict
                loss = loss_mse = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                metrics_to_log = {"train/mse": loss_mse.item()}
                if explicit_zero_prob:
                    loss_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mlm_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_zero_log_prob
                    metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
                if config.GEPC:
                    loss_gepc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc
                    metrics_to_log.update({"train/mvc": loss_gepc.item()})
                if config.GEPC and explicit_zero_prob:
                    loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                        output_dict["mvc_zero_probs"], target_values, masked_positions
                    )
                    loss = loss + loss_gepc_zero_log_prob
                    metrics_to_log.update(
                        {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                    )
                # print(config.dab_weight,config.ecs_thres)
                
                # print('loss_ecs: ',loss_ecs)
                if config.ecs_thres > 0:
                    loss_ecs = 10 * output_dict["loss_ecs"]
                    loss = loss + loss_ecs
                    metrics_to_log.update({"train/ecs": loss_ecs.item()})
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
                # print('loss dab: ',loss_dab)
                loss = loss + config.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            wandb.log(metrics_to_log)

            with torch.no_grad():
                mre = masked_relative_error(
                    output_dict["mlm_output"], target_values, masked_positions
                )

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_gepc += loss_gepc.item() if config.GEPC else 0.0
            total_error += mre.item()
            # total_dab += loss_dab.item()
            # total_ecs += loss_ecs.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_mse = total_mse / log_interval
                cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
                cur_error = total_error / log_interval
                # cur_ecs = total_ecs / log_interval 
                # cur_dab = total_dab / log_interval
                # ppl = math.exp(cur_loss)
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                    + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
                )
                # print(f'ecs loss: {cur_ecs}, dab loss: {cur_dab}')
                # print(
                #     f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | 
                #     lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | 
                #     loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |
                #     gepc {cur_gepc:5.2f} | ecs {cur_ecs:5.2f} | dab {cur_dab:5.2f}"
                # )
                total_loss = 0
                total_mse = 0
                total_gepc = 0
                total_error = 0
                start_time = time.time()


    def define_wandb_metrcis():
        wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
        wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
        wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
        wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
        wandb.define_metric("test/avg_bio", summary="max")


    def evaluate(model: nn.Module, loader: DataLoader) -> float:
        """
        Evaluate the model on the evaluation data.
        """
        model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        with torch.no_grad():
            for batch_data in loader:
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)

                src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
                with torch.cuda.amp.autocast(enabled=config.amp):
                    output_dict = model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        # batch_labels=batch_labels if DSBN else None,
                        batch_labels=batch_labels,
                    )
                    output_values = output_dict["mlm_output"]

                    masked_positions = input_values.eq(mask_value)
                    loss = criterion(output_values, target_values, masked_positions)
                    loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

                total_loss += loss.item() * len(input_gene_ids)
                total_error += masked_relative_error(
                    output_values, target_values, masked_positions
                ).item() * len(input_gene_ids)
                total_dab += loss_dab.item() * len(input_gene_ids)
                total_num += len(input_gene_ids)

        wandb.log(
            {
                "valid/mse": total_loss / total_num,
                "valid/mre": total_error / total_num,
                "valid/dab": total_dab / total_num,
                "valid/sum_mse_dab": (total_loss + config.dab_weight * total_dab)
                / total_num,
                "epoch": epoch,
            },
        )

        return total_loss / total_num, total_error / total_num


    def eval_testdata(
        model: nn.Module,
        adata_t: AnnData,
        include_types: List[str] = ["cls"],
    ) -> Optional[Dict]:
        """evaluate the model on test dataset of adata_t"""
        model.eval()

        # copy adata_t to avoid reuse previously computed results stored in adata_t
        adata_t = adata_t.copy()

        all_counts = (
            adata_t.layers[input_layer_key].A
            if issparse(adata_t.layers[input_layer_key])
            else adata_t.layers[input_layer_key]
        )

        celltypes_labels = adata_t.obs["celltype"].tolist()
        celltypes_labels = np.array(celltypes_labels)

        batch_ids = adata_t.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)

        # Evaluate cls cell embeddings
        if "cls" in include_types:
            logger.info("Evaluating cls cell embeddings")
            tokenized_all = tokenize_and_pad_batch(
                all_counts,
                gene_ids,
                max_len=max_seq_len,
                vocab=vocab,
                pad_token=pad_token,
                pad_value=pad_value,
                append_cls=True,  # append <cls> token at the beginning
                include_zero_gene=True,
            )
            all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
            src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
                cell_embeddings = model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=config.batch_size,
                    # batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
                    batch_labels=torch.from_numpy(batch_ids).long(),
                    time_step=0,
                    return_np=True,
                )
            cell_embeddings = cell_embeddings / np.linalg.norm(
                cell_embeddings, axis=1, keepdims=True
            )

            adata_t.obsm["X_scGPT"] = cell_embeddings

            results = {}
            try:
                results = eval_scib_metrics(adata_t)
            except Exception as e:
                traceback.print_exc()
                logger.error(e)

            sc.pp.neighbors(adata_t, use_rep="X_scGPT")
            sc.tl.umap(adata_t, min_dist=0.3)
            fig = sc.pl.umap(
                adata_t,
                color=["str_batch"],
                title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
                frameon=False,
                return_fig=True,
                show=False,
            )

            results["batch_umap"] = fig

            sc.pp.neighbors(adata_t, use_rep="X_scGPT")
            sc.tl.umap(adata_t, min_dist=0.3)
            fig = sc.pl.umap(
                adata_t,
                color=["celltype"],
                title=[
                    f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
                ],
                frameon=False,
                return_fig=True,
                show=False,
            )

            results["celltype_umap"] = fig

        if len(include_types) == 1:
            return results, cell_embeddings
        
    best_val_loss = float("inf")
    best_avg_bio = 0.0
    best_model = None
    define_wandb_metrcis()

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=config.batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )

        if config.do_train:
            train(
                model,
                loader=train_loader,
            )
        val_loss, val_mre = evaluate(
            model,
            loader=valid_loader,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        if epoch == config.epochs:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

            # eval on testdata
            results, embs_refer = eval_testdata(
                best_model,
                adata_t=adata_sorted if per_seq_batch_sample else adata_train,
                include_types=["cls"],
            )
            results_test, embs_query = eval_testdata(
                best_model,
                adata_t=adata_sorted_test if per_seq_batch_sample else adata_test,
                include_types=["cls"],
            )
            refer_df = pd.DataFrame(embs_refer, index=adata_sorted.obs_names)
            query_df = pd.DataFrame(embs_query, index=adata_sorted_test.obs_names)
            refer_df.to_csv(save_dir / "rna_refer.csv", header=False)
            query_df.to_csv(save_dir / "rna_query.csv", header=False)

            adata = sc.read(args.input_data)
            df_combined = pd.concat([refer_df, query_df],axis=0)
            df_comb_reindex = df_combined.reindex(adata.obs.index)

            np.save(save_dir / 'embeds_scgpt.npy',df_comb_reindex.values)
            adata.obsm['X_scGPT'] = df_comb_reindex.values.copy()
            adata.obs['map'] = 'reference'
            adata.obs.loc[adata.obs['tech']=='smartseq2','map'] = 'query'

            sc.pp.neighbors(adata, use_rep="X_scGPT")
            sc.tl.umap(adata, min_dist=0.3)
            fig, axes = plt.subplots(3, 1, figsize=(4, 12))
            # 第一个UMAP图，color = "celltype"
            sc.pl.umap(adata, color=["celltype"], ax=axes[0], show=False)
            axes[0].set_title("UMAP for Celltype")

            # 第二个UMAP图，color = "map"
            sc.pl.umap(adata, color=["map"], ax=axes[1], show=False)
            axes[1].set_title("UMAP for Map")

            # 第三个UMAP图，color = "tech"
            sc.pl.umap(adata, color=["tech"], ax=axes[2], show=False)
            axes[2].set_title("UMAP for Tech")
            results["batch_umap"].savefig(
                save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
            )

            results["celltype_umap"].savefig(
                save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
            )
            metrics_to_log = {"test/" + k: v for k, v in results.items()}
            metrics_to_log["test/batch_umap"] = wandb.Image(
                str(save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
                caption=f"celltype avg_bio epoch {best_model_epoch}",
            )

            metrics_to_log["test/celltype_umap"] = wandb.Image(
                str(save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
                caption=f"celltype avg_bio epoch {best_model_epoch}",
            )
            metrics_to_log["test/best_model_epoch"] = best_model_epoch
            wandb.log(metrics_to_log)
            wandb.log({"avg_bio": results.get("avg_bio", 0.0)})

        scheduler.step()

    torch.save(best_model.state_dict(), save_dir / "best_model.pt")

if __name__ == "__main__":
    main(parse_args())