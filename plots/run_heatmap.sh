#!/bin/bash

cd ./plots
echo "Creating synth_clean_score.csv ..."
ipython -c "%run plot_correlation_synth_clean.ipynb"

echo "Creating synth_cmeasures_nn.csv ..."
ipython -c "%run save_cmeasures_synth.ipynb"

echo "Creating heatmap ..."
ipython -c "%run heatmap_synth.ipynb"
