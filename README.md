# Lab 2: Fine-tuning an LLM

## 🖥️ [TODO: Click me to see the website]() ✨

This README contains all information on the second lab where an LLM is fine-tuned with the FineTome dataset.
For practical information, please scroll to the botton.

## Overview

Below is a table showing the LLMs we fine-tuned.
All LLMs were pre-trained by Unsloth and modified using the `unsloth` module.

| LLM | Parameters | Epochs | GPU | Training Time | Link |
| -- | -- | -- | -- | -- | -- |
| Llama 3.2 Instruct | 1B | 1 | RTX 2060S | ~9 hours | [Model](https://huggingface.co/Taiwar/llama-3.2-1b-instruct-lora_model-1epoch) |
| Llama 3.2 Instruct | 3B | 1 | RTX 5000 | ~18 hours | **TODO** |

It should be noted the 1B model was trained on a Windows machine, and the 3B model in Paperspace.

## Way of Working

In this project, both contributed equally.
The task breakdown is presented below.

| Task | Person | Comments
| -- | -- | -- |
| Training 1B | Jonas | Performed locally on the Windows machine |
| Training 3B | Paul | On Paperspace Gradient |
| Inference | Jonas | Uses Huggingface |
| UI | Paul | Experimented with the speech synthesis API |

## Training 

We investigated the training times a little and found the following results. Note that for many of these, especially the longer ones, we did not complete training.

### RTX 2060S

| Model/Parameters | Steps/min | Epochs | Time |
| -- | -- | -- | -- |
| Llama 3.2 Instruct 1B | 24 | 1 | ~9h |
| Llama 3.2 Instruct 1B | 24 | 3 | ~26h |
| Llama 3.2 Instruct 3B | ? | 3 | ~64h |

### RTX 3070

| Model/Parameters | Steps/min | Epochs | Time |
| -- | -- | -- | -- |
| Llama 3.2 Instruct 1B | 26 | 1 | ~9h |

### Quadro RTX 5000

| Model/Parameters | Steps/min | Epochs | Time |
| -- | -- | -- | -- |
| Llama 3.2 Instruct 3B | 3 | 1 | ~18h |

## Improvement

In order to improve the performance, we list both model-centric approaches and data-centric approaches.

### Model-centric approaches
**TODO: EXPLAIN.**

### Data-centric approaches
**TODO: EXPLAIN.**

## Technical Setup

This section outlines the setup steps for the both the 1B model (trained on Windows) and 3B model (trained in the cloud on Paperspace Gradient). Also outlined is deployment to Modal.

### Windows
- Python 3.10
- Dependency specifics for local install (given CUDA-capable GPU, CUDA 12.1 and win 11)
  - unsloth conda env via `conda create --name unsloth_env python=3.10 pytorch-cuda=12.1 pytorch cudatoolkit -c pytorch -c nvidia -y`
  - triton install on windows e.g via https://github.com/woct0rdho/triton-windows
  - unsloth install `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"`
  - install xformers via their repo instructions https://github.com/facebookresearch/xformers
  - install other unsloth deps `pip install --no-deps trl peft accelerate bitsandbytes`
  - fix torch versions with `conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 -c pytorch`
  - fix progress bars with `conda install -c conda-forge ipywidgets`
  - See and potentially use `spec-file.txt` (seemingly does not include pip installs)
- Hugging face token with write access saved into `notebooks/.hftoken`

### Paperspace

Use the Huggingface Transformers environment as a base template, and then run the following script to complete the setup.
```sh
#!/bin/bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
pip install "unsloth[cu118-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install trl==0.8.2
pip install deepspeed==0.14.4
pip install unsloth_zoo==2024.11.8
pip install --upgrade bitsandbytes
```

Write the Hugging face token to `notebooks/token.txt` (Jupyter does not allow for `.hftoken`).

### Modal Deployment
The `deployment` folder contains two scripts for setting up a model server on Modal
1. `modal run deployment/modal_download_model.py` will download a specified model from HF and store it in a shared volume
2. `modal run deployment/modal_inference.py` will start an OpenAPI compatible fastapi model server with the specified model with GPU acceleration
