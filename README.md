# Adversarial Label Flipping Attack (ALFA) and Detect Using Complex Measures

## Install dependencies

Virtual environment is created using `Python 3.9.13` with `PyTorch 1.12.1 (CUDA=11.6)`.

Code was tested on Ubuntu 20.04.5 LTS.

```bash
# Create virtual environment
python3.9 -m venv venv
source ./venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install from requirements.txt
pip install -r requirements.txt

# Install PyTorch
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install local package
pip install .

```
