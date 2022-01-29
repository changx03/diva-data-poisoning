#!/bin/bash

# echo "Step 1: Preprocessing ===================================================="
# cd ./experiments/real
# ipython -c "%run Step1_Preprocessing.ipynb"

# echo "Step 1: Train-Test split ================================================="
# cd ../../
# python ./experiments/real/Step1_TrainTestSplit.py -f "data/standard" -o "data/real"

DATASETS=("abalone_subset_std" "australian_std" "banknote_std" "breastcancer_std" "cmc_std" "htru2_subset_std" "phoneme_subset_std" "ringnorm_subset_std" "texture_subset_std" "yeast_subset_std")

echo "=========================================================================="
echo "Running Random Flip on SVM"
for DATA in "${DATASETS[@]}";
do
    echo "Step 2: Running Random Flip SVM =========================================="
    python ./experiments/real/Step2_RandomNoise.py -f "data/real" -d $DATA

    echo "Step 3: Computing C-Measures on Random Flip SVM =========================="
    Rscript ./experiments/real/Step3_CMeasure.R "./data/real/rand_svm/" "./results/real/rand_svm/${DATA}_cmeasure_rand_svm.csv" $DATA
done
# echo "Step 4: Create database =================================================="


# echo "=========================================================================="
# echo "Running FALFA on NN"
# for DATA in "${DATASETS[@]}";
# do
#     echo "Step 2: Running FALFA on NN =============================================="
#     python ./experiments/real/Step2_FALFA_NN.py -f "data/real" -d $DATA

#     echo "Step 3: Computing C-Measures on FALFA NN ================================="
#     Rscript ./experiments/real/Step3_CMeasure.R "./data/real/falfa_nn/" "./results/real/falfa_nn/${DATA}_cmeasure_falfa_nn.csv" $DATA
# done

# echo "=========================================================================="
# echo "Running SecML Poisoning SVM Attack"
# for DATA in "${DATASETS[@]}";
# do
#     echo "Step 2: Running SecML Poisoning SVM ======================================"
#     python ./experiments/real/Step2_SecMLPoisoningSVM.py -f "data/real" -d $DATA

#     echo "Step 3: Computing C-Measures on SecML Poisoning SVM ======================"
#     Rscript ./experiments/real/Step3_CMeasure.R "./data/real/poison_svm/" "./results/real/poison_svm/${DATA}_cmeasure_poison_svm.csv" $DATA
# done

# echo "=========================================================================="
# echo "Running ALFA SVM Attack"
# for DATA in "${DATASETS[@]}";
# do
#     echo "Step 2: Running ALFA SVM ================================================="
#     python ./experiments/real/Step2_ALFA_SVM.py -f "data/real" -d $DATA

#     echo "Step 3: Computing C-Measures on ALFA SVM ================================="
#     Rscript ./experiments/real/Step3_CMeasure.R "./data/real/alfa_svm/" "./results/real/alfa_svm/${DATA}_cmeasure_alfa_svm.csv" $DATA
# done
