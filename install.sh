#!/bin/bash

echo "This script only works on Linux with a CUDA enabled GPU!"
python3 -m pip ./venv
source ./venv/bin/activate
pip install --upgrade pip wheel
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r ./requirements.txt
pip install --upgrade .
