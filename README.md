# Lab 2: Fine-tuning an LLM

## 🖥️ [TODO: Click me to see the website]() ✨

This README contains all information on the second lab where an LLM is fine-tuned with the FineTome dataset.
For practical information, please scroll to the botton.

## Overview

Below is a table showing the LLMs we fine-tuned.
All LLMs were pre-trained by Unsloth and modified using the `unsloth` module.

| LLM | Parameters | Epochs | GPU | Training Time | Link |
| -- | -- | -- | -- | -- | -- |
| Llama 3.2 Instruct | 1B | 1 | RTX 2060S | ~9 hours | [Normal](https://huggingface.co/Taiwar/llama-3.2-1b-instruct-lora_model-1epoch), [Merged](https://huggingface.co/Taiwar/llama-3.2-1b-instruct-lora-1poch_merged16b) |
| Llama 3.2 Instruct | 3B | 1 | RTX 5000 | ~18 hours | [Normal](https://huggingface.co/Arraying/llama-3.2-3b-instruct-lora_model-1epoch), [Merged](https://huggingface.co/Arraying/llama-3.2-3b-instruct-lora-1poch_merged16b) |

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

## Fine Tuning Parameter Explanation
In PEFT, low-rank matrices are injected at every level of a transformer model.
The original model parameters are frozen and only the low-rank matrices are trained.

The trained low-rank matrices appear to be able to strongly emphasize certain patterns which the original model may have learned but did not express as strongly.

Sources: 
- [LoRA: Low-Rank Adaptation of Large Language Models](http://arxiv.org/abs/2106.09685)
- [Unsloth docs](https://docs.unsloth.ai/basics/lora-parameters-encyclopedia)

### PEFT Model Parameters

- `r = 16` Rank of the low-rank decomposition for factorizing weight matrices
  - Tradeoff between information **retention/expressiveness** and **computational load**
  - Expressed strong diminishing returns as `r` increases in LoRA paper (`r > 8` in GPT-3)
- `lora_alpha = 16` Scaling factor for the low-rank matrices contribution
  - Tradeoff between **convergence speed** and **stability/overfitting**
- `lora_dropout = 0` Dropout rate for the low-rank matrices (zeroing out elements)
  - Tradeoff between **regularization/preventing overfitting** and **training speed**
- `loftq_config = None` Whether to use [LoftQ](https://arxiv.org/abs/2310.08659), which is a quantization method for the backbone weights and LoRA initialization
  - Only use it if the pretrained model is not already quantized
- `use_rslora = False` Whether to use Rank-Stabilized LoRA
  - Uses scaling factor proposed in [this paper](https://arxiv.org/abs/2312.03732) for `lora_alpha` to 
    potentially improve fine-tuning performance (especially for larger `r` values)
  - Increases **computational load** during training but does not affect inference performance
- `use_gradient_checkpointing = "unsloth"` Strategy for only storing a subset of gradients during backpropagation, non-checkpointed layers need to be recomputed based on the stored gradients
  - Tradeoff between **memory usage** and **computation**
- `target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` Which transformer layers to apply the low-rank matrices to
- `bias = "none"` Potentially which biases to train (if any), but appears not supported in the current unsloth implementation

### Training Parameters
- `gradient_accumulation_steps = 4` Number of steps to accumulate gradients before performing a backpropagation update.
  Effectively **increases the batch size** without requiring as much additional memory.
  - Tradeoff between **training stability/convergence speed** and **memory usage**
- `weight_decay = 0.01` Regularization technique which applied a small penalty to the model's weights to prevent overfitting (discourages large weights).
  - Tradeoff between **preventing overfitting** and potentially **convergence speed**
- `learning_rate = 2e-4` Rate at which the model's weights are updated during training.
  - Tradeoff between **convergence speed** and **stability**
- `warmup_steps = 5` Steps over which the learning rate increases linearly from 0 to the specified learning rate.
  **Improves stability** and works against catastrophic forgetting.
- `lr_scheduler_type = "linear"` How to adjust learning rate over time.
- `per_device_train_batch_size = 2` Number of samples per batch per GPU.
  - Tradeoff between **training stability/convergence speed** and **memory usage**

## Improvement

In order to improve the performance, we list both model-centric approaches and data-centric approaches.

### Model-centric approaches
- Tune hyperparameters
  - Explore using Rank-Stabilized LoRA (and higher `r` values depending on computational increase)
  - Explore using higher `lora_alpha` values with `lora_dropout` to prevent overfitting
  - Tune `gradient_accumulation_steps` and `per_device_train_batch_size` to find the largest possible batch size for the given hardware
- Try using LoftQ on a non-quantized model instead of fine-tuning a quantized model
  - May improve model performance, but also may increase training time
- Experiment with different models & sizes\
  → Overall tradeoff between **model size** (**affects model's ability to tackle complex tasks**), **quantization** (affects **accuracy**), and **batch size** (and thus **training time**)

### Data-centric approaches
- Use larger fine-tuning datasets
  - More data can help the model generalize better
- Improve data quality
  - Look for well curated datasets
  - Look data more representative of the target domain

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
