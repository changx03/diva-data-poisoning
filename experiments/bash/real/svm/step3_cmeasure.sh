#!bin/bash

# Compute C-Measures for each data set and save results as CSV file.

Rscript ./experiments/real/svm/step3_cmeasure.R "./data/output/train" "./data/output/alfa" cmeasure_real_svm
