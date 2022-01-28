#!/bin/bash

# Generate synthetic data
python ./experiments/synth/Step1_GenDataByDifficulty.py -n 150 -f "synth"

python ./experiments/synth/Step1_GenDataByNoise.py -n 50 -f "synth_noisy"
