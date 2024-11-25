# Lab 2: Fine-tuning an LLM

## Pre-requisites
- Python 3.10
- Dependency specifics for local install (given CUDA-capable GPU, CUDA 12.1 and win 11)
  - unsloth conda env via `conda create --name unsloth_env python=3.10 pytorch-cuda=12.1 pytorch cudatoolkit -c pytorch -c nvidia -y`
  - triton install on windows e.g via https://github.com/woct0rdho/triton-windows
  - unsloth install `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
  - install xformers via their repo instructions https://github.com/facebookresearch/xformers
  - install unsloth `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
  - install other unsloth deps `pip install --no-deps trl peft accelerate bitsandbytes`
  - fix torch versions with `conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch
  - fix progress bars with `conda install -c conda-forge ipywidgets`
  - See and potentially use `spec-file.txt` (seemingly does not include pip installs)
- Hugging face token with write access saved into `notebooks/.hftoken`

## Training Notes
On RTX 2060 SUPER
- Llama-3.2-3B-Instruct
  - ~9 steps per minute and ~64 h for 3 epochs
- Llama-3.2-1B-Instruct
  - ~24 steps per minute and ~26 h for 3 epochs

## Improving Performance
TODO

### Model-centric approaches
TODO

### Data-centric approaches
TODO