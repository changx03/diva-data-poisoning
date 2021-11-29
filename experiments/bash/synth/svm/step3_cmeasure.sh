#!bin/bash

# Compute C-Measures for each data set and save results as CSV file.

Rscript ./experiments/synth/svm/step3_cmeasure.R "./data/synth/" alfa cmeasure_synth_svm
