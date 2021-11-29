#!bin/bash

# Compute C-Measures for each data set and save results as CSV file.

Rscript ./experiments/synth/svm/step3_cmeasure.R "./data/synth/" alfa_nn cmeasure_synth_nn
