#!/bin/bash

rm -rvf ./data/real/poison_svm
rm -rvf ./results/real/poison_svm
find ./results -type f -name "*poison_svm*" -exec rm -fv {} \;
