#!/bin/bash

conda create -n env1 python=3.8 -y
conda activate env1
pip install torch==2.2.1 torchvision==0.17.1
pip install scanpy==1.9.8
pip install scimilarity