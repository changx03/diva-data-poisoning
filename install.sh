#!/bin/bash

OSNAME=$(uname)
if [[ $OSNAME != "Linux" ]]; then 
    echo "This script only works on Linux with a CUDA enabled GPU!"
    exit 1
fi

echo "Creating virtual environment in ./venv"
python3 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install secml
pip install -r ./requirements.txt
pip install --upgrade .
python ./demo/cuda_test.py
python ./demo/check_packages.py
Rscript ./demo/check_packages.R
exit 0