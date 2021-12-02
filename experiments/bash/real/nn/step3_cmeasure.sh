#!bin/bash

# Compute C-Measures for real datasets and save results as CSV file.
# Parameters:
# 1: data path. 2: output path. 3: output name

echo "C-Measure for clean training data should be completed in SVM"
mkdir ./results/real_nn
Rscript ./experiments/synth/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/" "real_nn_poison"
