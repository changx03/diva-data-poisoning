#!/bin/bash

echo "Running on ALFA SVM..."
python ./experiments/baseline/knndefense_by_noise.py -i "./results/synth_noisy/synth_alfa_svm_score.csv" -o "./results/synth_noisy/baseline/synth_alfa_svm_knndefense.csv"

echo "Running on FALFA NN..."
python ./experiments/baseline/knndefense_by_noise.py -i "./results/synth_noisy/synth_falfa_nn_score.csv" -o "./results/synth_noisy/baseline/synth_falfa_nn_knndefense.csv"