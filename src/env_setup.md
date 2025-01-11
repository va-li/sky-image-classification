- sudo apt install python3.11
- sudo apt install python3.11-venv
- sudo apt install nvidia-cuda-toolkit

- python3.11 -m venv env
- source env/bin/activate
- pip install --upgrade pip

- pip install numpy pandas scikit-learn scikit-image tqdm matplotlib tabulate
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


```
import torch
torch.cuda.is_available()
```
