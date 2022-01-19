#!bin/bash

# Compute C-Measures for real datasets and save results as CSV file.
# Parameters:
# 1: data path. 2: output path. 3: output name
mkdir ./results/synth_svm

echo "For clean data..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/" "./results/synth_svm/" "synth_train_clean"

echo "For poisoned data..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/alfa/" "./results/synth_svm/" "synth_svm_poison"
