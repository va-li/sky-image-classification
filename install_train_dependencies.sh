#!/bin/bash

# Install system dependencies
sudo apt install python3.11
sudo apt install python3.11-venv
sudo apt install nvidia-cuda-toolkit

# Create virtual environment here
python3.11 -m venv ./venv
source ./venv/bin/activate

# Upgrade pip and install python dependencies
pip install --upgrade pip
pip install numpy pandas scikit-learn scikit-image tqdm matplotlib tabulate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
