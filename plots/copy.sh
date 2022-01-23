#!/bin/bash

cp ./results/synth_cor_acc_test_clean.pdf ../overleaf_cmeasures/images/
cp ./results/synth_cor_acc_test_falfa_0.20.pdf ../overleaf_cmeasures/images/
cp ./results/synth_cor_rate_falfa.pdf ../overleaf_cmeasures/images/
cp ./results/abalone_falfa_line.pdf ../overleaf_cmeasures/images/
cp ./results/texture_falfa_line.pdf ../overleaf_cmeasures/images/
cp ./results/roc_real_repeated.pdf ../overleaf_cmeasures/images/
cp ./results/roc_synth.pdf ../overleaf_cmeasures/images/
cp ./results/fake_acc.pdf ../overleaf_cmeasures/images/
cp ./results/flfa_acc.pdf ../overleaf_cmeasures/images/

cd ~/workspace/overleaf_cmeasures
git pull
git add -A
git commit -m "update plots"
git push
