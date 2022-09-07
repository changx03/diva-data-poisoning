#!/bin/bash

# cd ./plots_v2
# echo "Generate plots..."
# ipython -c "%run plot_corr.ipynb"
# ipython -c "%run plot_fake_acc.ipynb"
# ipython -c "%run plot_heatmap_3.ipynb"
# ipython -c "%run plot_real_line.ipynb"
# ipython -c "%run plot_roc.ipynb"

# cd ..
echo "#########################################################################"
echo "Copy plots..."
cp ./results_plot/cm_line.pdf ../overleaf_cmeasures/images/
cp ./results_plot/fake_acc.pdf ../overleaf_cmeasures/images/
cp ./results_plot/flfa_acc.pdf ../overleaf_cmeasures/images/
cp ./results_plot/roc.pdf ../overleaf_cmeasures/images/
cp ./results_plot/synth_corr.pdf ../overleaf_cmeasures/images/
cp ./results_plot/synth_heatmap.svg ../overleaf_cmeasures/images/


echo "#########################################################################"
echo "Commit to GitHub..."
cd ~/workspace/overleaf_cmeasures
git pull
git add -A
git commit -m "update plots"
git push
