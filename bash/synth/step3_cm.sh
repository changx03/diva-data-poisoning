#!/bin/bash

echo "Computing C-Measures on FALFA NN..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/falfa_nn/" "./results/synth/falfa_nn/" "synth_cmeasure_falfa_nn"

echo "Computing C-Measures on ALFA SVM..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/alfa_svm/" "./results/synth/alfa_svm/" "synth_cmeasure_alfa_svm"

echo "Computing C-Measures on FALFA NN (Noise)..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth_noisy/falfa_nn/" "./results/synth_noisy/falfa_nn/" "synth_noisy_cmeasure_falfa_nn"

echo "Computing C-Measures on ALFA SVM (Noise)..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth_noisy/alfa_svm/" "./results/synth_noisy/alfa_svm/" "synth_noisy_cmeasure_alfa_svm"

echo "Computing C-Measures on noisy data..."
Rscript ./experiments/synth/step3_cmeasure.R "./data/synth/rand/" "./results/synth/rand/" "synth_cmeasure_rand"
