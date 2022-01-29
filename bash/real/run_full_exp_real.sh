#!/bin/bash

echo "Step 1: Preprocessing ===================================================="
cd ./experiments/real
ipython -c "%run Step1_Preprocessing.ipynb"

echo "Step 1: Train-Test split ================================================="
cd ../../
python ./experiments/real/Step1_TrainTestSplit.py -f "data/standard" -o "data/real"

DATASETS=("abalone_subset_std" "australian_std" "banknote_std" "breastcancer_std" "cmc_std" "htru2_subset_std" "phoneme_subset_std" "ringnorm_subset_std" "texture_subset_std" "yeast_subset_std")

for DATA in "${DATASETS[@]}";
do
    echo "Step 2: Running SecML Poisoning SVM ======================================"
    python ./experiments/real/Step2_SecMLPoisoningSVM.py -f "data/real" -o "results/real" -d $DATA

    echo "Step 3: Computing C-Measures on SecML Poisoning SVM ======================"
    Rscript ./experiments/real/Step3_CMeasure.R "./data/real/poison_svm/" "./results/poison_svm/$DATA\_cmeasure_poison_svm_.csv" $DATA
done
