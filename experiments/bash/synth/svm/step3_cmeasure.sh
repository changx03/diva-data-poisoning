#!bin/bash

# Compute C-Measures for each data set and save results as CSV file.
# 1: clean path. 2: poison path. 3: output path. 4: output name
mkdir ./results/synth_svm
Rscript ./experiments/synth/svm/step3_cmeasure.R "./data/synth/" "./data/synth/alfa/" "./results/synth_svm/" "synth_svm"
