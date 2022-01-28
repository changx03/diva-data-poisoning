#!/bin/bash

cd ./experiments/real
echo "Data preprocessing..."
ipython -c "%run Step1_Preprocessing.ipynb"

cd ../../
python ./experiments/real/Step1_TrainTestSplit.py -f "data/standard" -o "data/real"
