#!bin/bash

# Compute C-Measures for real datasets and save results as CSV file.
# Parameters:
# 1: data path. 2: output path. 4: output name
mkdir ./results/real_svm
echo "For clean data..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/output/train/" "./results/real_svm/" "real_train_clean"

echo "For poisoned data..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/" "real_svm_poison"
