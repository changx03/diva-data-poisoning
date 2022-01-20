#!bin/bash

# Compute C-Measures for real datasets and save results as CSV file.
# Parameters:
# 1: data path. 2: output path. 3: data name

echo "C-Measure for clean training data should be completed in SVM"
mkdir ./results/real_random
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/real_random/abalone_random_poison.csv" "abalone"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/real_random/australian_random_poison.csv" "australian"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/real_random/banknote_random_poison.csv" "banknote"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/real_random/breastcancer_random_poison.csv" "breastcancer"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/real_random/cmc_random_poison.csv" "cmc"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/real_random/htru2_random_poison.csv" "htru2"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/real_random/phoneme_random_poison.csv" "phoneme"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/real_random/ringnorm_random_poison.csv" "ringnorm"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/real_random/texture_random_poison.csv" "texture"
Rscript ./experiments/real/step3_cmeasure.R "./data/output/random/" "./results/real_random/yeast_random_poison.csv" "yeast"
