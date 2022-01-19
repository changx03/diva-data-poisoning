#!bin/bash

# Compute C-Measures for real datasets and save results as CSV file.
# Parameters:
# 1: data path. 2: output path. 3: data name

echo "C-Measure for clean training data should be completed in SVM"
mkdir ./results/real_nn
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/abalone_nn_poison.csv" "abalone"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/australian_nn_poison.csv" "australian"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/banknote_nn_poison.csv" "banknote"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/breastcancer_nn_poison.csv" "breastcancer"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/cmc_nn_poison.csv" "cmc"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/htru2_nn_poison.csv" "htru2"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/phoneme_nn_poison.csv" "phoneme"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/ringnorm_nn_poison.csv" "ringnorm"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/texture_nn_poison.csv" "texture"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa_nn/" "./results/real_nn/yeast_nn_poison.csv" "yeast"
