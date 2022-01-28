#!bin/bash

# Compute C-Measures for real datasets and save results as CSV file.
# Parameters:
# 1: data path. 2: output path. 4: output name
echo "For poisoned data..."
mkdir ./results/real_svm
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/abalone_svm_poison.csv" "abalone"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/australian_svm_poison.csv" "australian"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/banknote_svm_poison.csv" "banknote"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/breastcancer_svm_poison.csv" "breastcancer"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/cmc_svm_poison.csv" "cmc"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/htru2_svm_poison.csv" "htru2"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/phoneme_svm_poison.csv" "phoneme"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/ringnorm_svm_poison.csv" "ringnorm"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/texture_svm_poison.csv" "texture"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/alfa/" "./results/real_svm/yeast_svm_poison.csv" "yeast"
