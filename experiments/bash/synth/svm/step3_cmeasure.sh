#!bin/bash

# Compute C-Measures for each data set and save results as CSV file.
# 1: clean path. 2: poison path. 3: output path. 4: output name
Rscript ./experiments/synth/svm/step3_cmeasure.R "./data/synth/" "./data/synth/alfa/" "./results/" "synth_svm"
