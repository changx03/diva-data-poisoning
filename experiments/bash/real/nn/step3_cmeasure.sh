#!bin/bash

# Compute C-Measures for real datasets and save results as CSV file.
# Parameters:
# 1: clean path. 2: poison path. 3: output path. 4: output name
mkdir ./results/real_nn
Rscript ./experiments/real/svm/step3_cmeasure.R "./data/output/train/" "./data/output/alfa_nn/" "./results/real_nn/" "real_nn"
