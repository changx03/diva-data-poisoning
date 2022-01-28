#!/bin/bash

echo "Computing C-Measures on clean data..."
bash ./experiments/bash/real/step3_cm_clean.sh

echo "Computing C-Measures on random noise data..."
bash ./experiments/bash/real/step3_cm_rand.sh

cd ./plots
echo "Save scores..."
ipython -c "%run save_scores.ipynb"

echo "Save C-Measures..."
ipython -c "%run save_cmeasures_real.ipynb"

echo "Save ROC plots..."
ipython -c "%run plot_roc_real.ipynb"
