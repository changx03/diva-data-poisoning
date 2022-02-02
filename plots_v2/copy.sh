#!/bin/bash

cp ./results_plot/flfa_acc.pdf ../overleaf_cmeasures/images/
cp ./results_plot/synth_corr.pdf ../overleaf_cmeasures/images/
cp ./results_plot/australian_std_line.pdf ../overleaf_cmeasures/images/
cp ./results_plot/texture_subset_std_line.pdf ../overleaf_cmeasures/images/

# cp ./results/roc_real_repeated.pdf ../overleaf_cmeasures/images/
# cp ./results/roc_synth.pdf ../overleaf_cmeasures/images/
# cp ./results/fake_acc.pdf ../overleaf_cmeasures/images/
# cp ./results/synth_heatmap_up3.svg ../overleaf_cmeasures/images/
# cp ./results/synth_heatmap_noise.svg ../overleaf_cmeasures/images/

cd ~/workspace/overleaf_cmeasures
git pull
git add -A
git commit -m "update plots"
git push
