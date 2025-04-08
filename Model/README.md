## Setup

``` bash
git clone https://github.com/marrb/DIP
cd Model
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
pip install -r requirements.txt
```

``` bash
vim /usr/local/lib/python3.10/dist-packages/diffusers/dynamic_modules_utils.py
```
Remove cached_download from imports and then replace with hf_hub_download in code.


``` bash
cd ..
git clone https://github.com/facebookresearch/xformers/
cd xformers
git submodule update --init --recursive
sudo apt-get install python3-dev
```

## Quickstart

``` bash
# Stage 1: Tuning to do model initialization.

# You can minimize the tuning epochs to speed up.
python3 run_tuning.py  --config="configs/rabbit-jump-tune.yaml"
```

``` bash
# Stage 2: Attention Control

# We develop a faster mode (1 min on V100):
python3 run_videop2p.py --config="configs/rabbit-jump-p2p.yaml" --fast

# The official mode (10 mins on V100, more stable):
python3 run_videop2p.py --config="configs/rabbit-jump-p2p.yaml"
```

Find your results in **Video-P2P/outputs/xxx/results**.