#!/bin/bash

cp ./results/synth_cor_acc_test_clean.pdf ../overleaf_cmeasures/images/
cp ./results/synth_cor_acc_test_falfa_0.20.pdf ../overleaf_cmeasures/images/
cp ./results/synth_cor_rate_falfa.pdf ../overleaf_cmeasures/images/

cd ~/worksapce/overleaf_cmeasures
git pull
git add -A
git commit -m "update plots"
git push
