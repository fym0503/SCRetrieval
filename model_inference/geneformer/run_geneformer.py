import os
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["NCCL_DEBUG"] = "INFO"
# imports
from collections import Counter
import datetime
import pickle
import subprocess
import seaborn as sns; sns.set()
from datasets import load_from_disk,concatenate_datasets
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments
from geneformer import DataCollatorForCellClassification
from utils.h5ad_to_dataset import data_preparation_geneformer,extract_emb_geneformer
import argparse
from pathlib import Path
import time
#from geneformer import EmbExtractor
def parse_args():
    parse = argparse.ArgumentParser(description='Cell Embedding Retrieval from data')  # 2、创建参数对象
    parse.add_argument('--input_adata', default=None, type=str, help='Input file path')  # 3、往参数对象添加参数
    parse.add_argument('--output_dir', default=None, type=str, help='Output file directory')
    parse.add_argument('--model_save_dir', default=None, type=str, help='Saved model directory')
    args = parse.parse_args()  # 4、解析参数对象获得解析对象
    return args

args = parse_args()
dataset_path , dataset_organ,adata = data_preparation_geneformer(args.input_adata)
original_index_list = list(adata.obs.index)
train_dataset=load_from_disk(dataset_path)
path = args.model_save_dir
obj = Path(path)
if (obj.exists()):
    start_cpu_time = time.process_time()
    extract_emb_geneformer(original_index_list,8,"{}".format(path),dataset_path,dataset_organ,args.output_dir,adata)
    execution_time = time.process_time() - start_cpu_time

    print(f"Embedding extraction time in seconds: {execution_time:.6f} seconds")
else:
    print("Please enter a correct path for a pretrained model.")