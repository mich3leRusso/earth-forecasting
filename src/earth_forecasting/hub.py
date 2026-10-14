"""Publish a trained LoRA adapter to a Hugging Face model repository.

The repository gets the adapter (best.pt, or last.pt if validation never beat the base model),
the exact config it was trained with, and a short model card. Set `hub.repo_id` in the config
and log in once with `hf auth login` (or set HF_TOKEN), then run `uv run earth configs/train.yaml steps=[publish]`.

Run `python -m earth_forecasting.hub` for a smoke test that builds the folder without uploading.
"""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from omegaconf import OmegaConf

from earth_forecasting.config import load_config

MODEL_CARD = """---
base_model: nvidia/stormscope-meteosat
tags: [weather, nowcasting, satellite, meteosat, lora, physicsnemo]
---

# {name}

LoRA adapter for NVIDIA's [StormScope-Meteosat](https://huggingface.co/nvidia/stormscope-meteosat),
fine-tuned to predict Meteosat Third Generation FCI imagery {step} minutes ahead over
latitudes {lat} and longitudes {lon}.

The base model predicts 10 minutes ahead from six frames 10 minutes apart. This adapter keeps
the base weights frozen and adds rank-{rank} LoRA matrices to the attention layers ({layers})
of blocks {blocks} in both diffusion experts{output_layer}, so the frames it reads and predicts are
{step} minutes apart.

Training windows start between {train_start} and {train_end}; validation between {val_start}
and {val_end}. NVIDIA trained the base model on November 2024 to May 2026, so these dates are
new to it.

## Files

- `{adapter}`: the adapter weights (physicsnemo `save_adapter` format).
- `train.yaml`: the exact configuration used for training.

## Using it

Load the base model with the same crop and time step, apply LoRA with the same settings, then
load the weights with `physicsnemo.experimental.peft.load_adapter`. The training code is in
[earth-forecasting](https://github.com/mich3leRusso/earth-forecasting), where `train.setup_model`
does the first two steps.

Contains modified EUMETSAT Meteosat FCI Level 1c data. The base model is released under the
OpenMDW-1.1 license; check it and EUMETSAT's data terms before reusing this adapter.
"""


def stage(cfg, folder: Path) -> Path:
    """Copy the adapter and config into `folder` and write the model card. Returns `folder`."""
    ckpt = Path(cfg.output.checkpoints)
    adapter = next((ckpt / n for n in ("best.pt", "last.pt") if (ckpt / n).exists()), None)
    if adapter is None:
        raise FileNotFoundError(f"no best.pt or last.pt in {ckpt}; train first")
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(adapter, folder / adapter.name)
    OmegaConf.save(cfg, folder / "train.yaml")
    (folder / "README.md").write_text(
        MODEL_CARD.format(
            name=cfg.hub.repo_id or "StormScope LoRA adapter",
            step=cfg.data.step_minutes,
            lat=list(cfg.data.region.lat),
            lon=list(cfg.data.region.lon),
            rank=cfg.lora.rank,
            layers=", ".join(cfg.lora.layers),
            blocks=", ".join(str(b) for b in cfg.lora.blocks),
            output_layer=", plus the final projection to the 16 channels" if cfg.lora.get("output_layer") else "",
            train_start=cfg.data.train[0],
            train_end=cfg.data.train[1],
            val_start=cfg.data.val[0],
            val_end=cfg.data.val[1],
            adapter=adapter.name,
        )
    )
    return folder


def publish(cfg) -> str:
    """Create the repo if needed (private by default) and upload the staged folder. Returns the repo URL."""
    if not cfg.hub.repo_id:
        raise ValueError("set hub.repo_id in the config, e.g. hub.repo_id=your-name/stormscope-ireland-lora")
    api = HfApi()
    url = api.create_repo(cfg.hub.repo_id, private=cfg.hub.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        api.upload_folder(repo_id=cfg.hub.repo_id, folder_path=stage(cfg, Path(d)), commit_message="Upload LoRA adapter")
    return str(url)


def run(cfg) -> None:
    """Step `publish`."""
    print(publish(cfg))


def smoke_test():
    """Stage a fake adapter: the folder must hold the adapter, the config and a filled-in model card."""
    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt"
        ckpt.mkdir()
        (ckpt / "last.pt").write_bytes(b"fake adapter")
        cfg = OmegaConf.merge(load_config(args=[]), {"output": {"checkpoints": str(ckpt)}, "hub": {"repo_id": "someone/test"}})
        folder = stage(cfg, Path(d) / "staged")
        assert sorted(p.name for p in folder.iterdir()) == ["README.md", "last.pt", "train.yaml"]
        card = (folder / "README.md").read_text()
        assert "someone/test" in card and "{" not in card, "model card has unfilled fields"
        assert OmegaConf.load(folder / "train.yaml") == cfg, "staged config differs from the training config"
    print("hub OK: adapter, config and model card staged (nothing uploaded)")


if __name__ == "__main__":
    smoke_test()
