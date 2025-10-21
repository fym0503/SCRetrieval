#!/bin/bash

conda create -n env5 python=3.9 -y 
conda activate env5
conda install cudatoolkit=11.7 -c pytorch -c nvidia
pip install -r env5_requirements.txt