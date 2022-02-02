#!/bin/bash

echo "Generate plots..."
cd ./plots_v2
ipython -c "%run plot_corr.ipynb"
ipython -c "%run plot_real_acc.ipynb"
ipython -c "%run plot_real_line.ipynb"
ipython -c "%run plot_real_roc_repeat.ipynb"
ipython -c "%run plot_synth_roc.ipynb"
ipython -c "%run plot_theoretical_acc.ipynb"

echo "#########################################################################"
echo "Copy plots..."
cd ..
cp ./results_plot/flfa_acc.pdf ../overleaf_cmeasures/images/
cp ./results_plot/synth_corr.pdf ../overleaf_cmeasures/images/
cp ./results_plot/australian_std_line.pdf ../overleaf_cmeasures/images/
cp ./results_plot/texture_subset_std_line.pdf ../overleaf_cmeasures/images/
cp ./results_plot/roc_real_repeated.pdf ../overleaf_cmeasures/images/
cp ./results_plot/roc_synth.pdf ../overleaf_cmeasures/images/
cp ./results_plot/fake_acc.pdf ../overleaf_cmeasures/images/

# cp ./results/synth_heatmap_up3.svg ../overleaf_cmeasures/images/
# cp ./results/synth_heatmap_noise.svg ../overleaf_cmeasures/images/

echo "#########################################################################"
echo "Commit to GitHub..."
cd ~/workspace/overleaf_cmeasures
git pull
git add -A
git commit -m "update plots"
git push
