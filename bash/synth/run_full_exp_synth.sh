#!/bin/bash

echo "By difficulty ============================================================"
echo "Generating synth data by difficulty..."
python ./experiments/synth/Step1_GenDataByDifficulty.py -n 3 -f "synth/full"
echo "Split synth data..."
python ./experiments/synth/Step1_TrainTestSplit.py -f "data/synth/full" -o "data/synth"

echo "Running FALFA on Neural Network classifier..."
python ./experiments/synth/Step2_FALFA_NN.py -f "./data/synth/" -o "./results/synth"
echo "Computing C-Measures on FALFA NN..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/falfa_nn/" "./results/synth/falfa_nn/" "synth_cmeasure_falfa_nn"

echo "Running ALFA on SVM classifier..."
python ./experiments/synth/Step2_ALFA_SVM.py -f "./data/synth/" -o "./results/synth"
echo "Computing C-Measures on ALFA SVM..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/alfa_svm/" "./results/synth/alfa_svm/" "synth_cmeasure_alfa_svm"

echo "Generating noise..."
python ./experiments/synth/Step2_RandomFlip.py -f "data/synth/"
echo "Computing C-Measures on noisy data..."
mkdir -p ./results/synth/rand/
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/rand/" "./results/synth/rand/" "synth_cmeasure_rand"


echo "By noise ================================================================="
echo "Generating synth data by noise rates..."
python ./experiments/synth/Step1_GenDataByNoise.py -n 1 -f "synth_noisy/full"
echo "Split synth data..."
python ./experiments/synth/Step1_TrainTestSplit.py -f "data/synth_noisy/full" -o "data/synth_noisy"

echo "Running FALFA on Neural Network classifier (Noise)..."
python ./experiments/synth/Step2_FALFA_NN.py -f "./data/synth_noisy/" -o "./results/synth_noisy"
echo "Computing C-Measures on FALFA NN (Noise)..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth_noisy/falfa_nn/" "./results/synth_noisy/falfa_nn/" "synth_noisy_cmeasure_falfa_nn"

echo "Running ALFA on SVM classifier (Noise)..."
python ./experiments/synth/Step2_ALFA_SVM.py -f "./data/synth_noisy/" -o "./results/synth_noisy"
echo "Computing C-Measures on ALFA SVM (Noise)..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth_noisy/alfa_svm/" "./results/synth_noisy/alfa_svm/" "synth_noisy_cmeasure_alfa_svm"
