#!bin/bash

# Compute C-Measure for Synthetic data trained on neural network models 
# 1: data path. 3: output path. 4: output name
mkdir ./results/synth_rand
echo "C-Measure for clean training data should be completed in SVM"
echo "For poisoned data..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/rand/" "./results/synth_rand/" "synth_rand"
