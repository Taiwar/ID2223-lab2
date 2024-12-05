# ---
# args: ["--force-download"]
# ---
import modal

MODELS_DIR = "/llamas"


volume = modal.Volume.from_name("llamas", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster with Rust
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


MINUTES = 60
HOURS = 60 * MINUTES


app = modal.App(
    image=image, secrets=[modal.Secret.from_name("huggingface-secret")]
)


@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, model_revision, force_download=False):
    from huggingface_hub import snapshot_download
    print(f"Downloading {model_name} at revision {model_revision}")

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        revision=model_revision,
        force_download=force_download,
    )

    print(f"Downloaded {model_name} at revision {model_revision}")
    volume.commit()


@app.local_entrypoint()
def main(
    force_download: bool = False,
):
    download_model.remote("Taiwar/llama-3.2-1b-instruct-lora-1poch_merged16b", "main", force_download)
    download_model.remote("Arraying/llama-3.2-3b-instruct-lora-1poch_merged16b", "main", force_download)