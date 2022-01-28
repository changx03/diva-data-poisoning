#!bin/bash

echo "For clean data..."
Rscript ./experiments/real/step3_cmeasure_clean.R "./data/output/train/" "./results/real_cm_clean.csv"
