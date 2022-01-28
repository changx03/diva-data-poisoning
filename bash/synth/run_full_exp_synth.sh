#!/bin/bash

echo "By difficulty ============================================================"
echo "Step 1: Generating synth data by difficulty..."
python ./experiments/synth/Step1_GenDataByDifficulty.py -n 3 -f "synth/full"
echo "Step 1: Split synth data..."
python ./experiments/synth/Step1_TrainTestSplit.py -f "data/synth/full" -o "data/synth"

echo "Step 2: Running FALFA on Neural Network classifier..."
python ./experiments/synth/Step2_FALFA_NN.py -f "./data/synth/" -o "./results/synth"
echo "Step 3: Computing C-Measures on FALFA NN..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/falfa_nn/" "./results/synth/falfa_nn/" "synth_cmeasure_falfa_nn"

echo "Step 2: Running ALFA on SVM classifier..."
python ./experiments/synth/Step2_ALFA_SVM.py -f "./data/synth/" -o "./results/synth"
echo "Step 3: Computing C-Measures on ALFA SVM..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/alfa_svm/" "./results/synth/alfa_svm/" "synth_cmeasure_alfa_svm"

echo "Step 2: Generating noise..."
python ./experiments/synth/Step2_RandomFlip.py -f "data/synth/"
echo "Step 3: Computing C-Measures on noisy data..."
mkdir -p ./results/synth/rand/
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/rand/" "./results/synth/rand/" "synth_cmeasure_rand"


echo "By noise ================================================================="
echo "Step 1: Generating synth data by noise rates..."
python ./experiments/synth/Step1_GenDataByNoise.py -n 1 -f "synth_noisy/full"
echo "Step 1: Split synth data..."
python ./experiments/synth/Step1_TrainTestSplit.py -f "data/synth_noisy/full" -o "data/synth_noisy"

echo "Step 2: Running FALFA on Neural Network classifier (Noise)..."
python ./experiments/synth/Step2_FALFA_NN.py -f "./data/synth_noisy/" -o "./results/synth_noisy"
echo "Step 3: Computing C-Measures on FALFA NN (Noise)..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth_noisy/falfa_nn/" "./results/synth_noisy/falfa_nn/" "synth_noisy_cmeasure_falfa_nn"

echo "Step 2: Running ALFA on SVM classifier (Noise)..."
python ./experiments/synth/Step2_ALFA_SVM.py -f "./data/synth_noisy/" -o "./results/synth_noisy"
echo "Step 3: Computing C-Measures on ALFA SVM (Noise)..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth_noisy/alfa_svm/" "./results/synth_noisy/alfa_svm/" "synth_noisy_cmeasure_alfa_svm"
