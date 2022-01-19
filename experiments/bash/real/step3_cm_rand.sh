#!bin/bash

# Compute C-Measures for real datasets and save results as CSV file.
# Parameters:
# 1: data path. 2: output path. 3: data name

echo "C-Measure for clean training data should be completed in SVM"
mkdir ./results/random
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/random/abalone_nn_poison.csv" "abalone"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/random/australian_nn_poison.csv" "australian"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/random/banknote_nn_poison.csv" "banknote"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/random/breastcancer_nn_poison.csv" "breastcancer"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/random/cmc_nn_poison.csv" "cmc"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/random/htru2_nn_poison.csv" "htru2"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/random/phoneme_nn_poison.csv" "phoneme"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/random/ringnorm_nn_poison.csv" "ringnorm"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/random/texture_nn_poison.csv" "texture"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/random/yeast_nn_poison.csv" "yeast"
