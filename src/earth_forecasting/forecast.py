"""Forecast with StormScope from real satellite frames: fetch the context, roll the model forward, save.

The six frames before `forecast.start` are downloaded like training data (or read from disk if
they are already there), then the model forecasts `forecast.steps` steps of `data.step_minutes`,
feeding each prediction back in. With an adapter (see evaluate.find_adapter) the fine-tuned model
runs; without one, the base model at the config's step.

The `forecast` step (see cli.py), e.g.
`uv run earth configs/evaluate.yaml steps=[forecast] forecast.start=2026-10-12T15:00`.
Run `python -m earth_forecasting.forecast` for a smoke test on synthetic frames (no network).
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from earth_forecasting.config import load_config, region_of
from earth_forecasting.data import FrameWindows, download_frames, synthetic_frame_shape, write_synthetic_frames
from earth_forecasting.evaluate import finetuned_model, rollout
from earth_forecasting.stormscope import N_INPUT


@torch.no_grad()
def run(cfg) -> Path:
    """Step `forecast`: forecast from cfg.forecast.start and save the predicted radiances (steps, C, H, W) as .npy."""
    step = np.timedelta64(cfg.data.step_minutes, "m")
    start = np.datetime64(cfg.forecast.start, "m")
    download_frames(start - (N_INPUT - 1) * step, start + step, cfg.data.dir, region_of(cfg), step)
    window = FrameWindows(start, start + step, cfg.data.dir, step, lead_steps=0)  # just the context
    if not len(window):
        raise FileNotFoundError(f"the {N_INPUT} frames up to {start} are not all available")
    frames, t = window[0]

    model, adapter = finetuned_model(cfg)
    if cfg.forecast.diffusion_steps:
        model.num_diffusion_steps = cfg.forecast.diffusion_steps
    print(f"forecasting {cfg.forecast.steps} x {cfg.data.step_minutes} min from {start} UTC with {adapter or 'the base model'}")

    radiances = model.denormalize(rollout(model, frames, t, cfg.forecast.steps)).cpu().numpy()
    path = Path(cfg.forecast.output) / f"forecast_{start}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, radiances)
    print(f"saved {radiances.shape} (steps, channels, y, x) to {path}")
    return path


def smoke_test():
    """Two steps with the base model from 6 synthetic frames already on disk, so nothing is downloaded."""
    cfg = load_config("evaluate.yaml", args=[])
    shape = synthetic_frame_shape(cfg)
    with tempfile.TemporaryDirectory() as d:
        write_synthetic_frames(d, "2026-07-01T00", n_frames=N_INPUT, shape=shape)
        cfg = OmegaConf.merge(cfg, {
            "data": {"dir": d},
            "adapter": None,
            "output": {"checkpoints": f"{d}/none"},  # no adapter anywhere: base model
            "forecast": {"start": "2026-07-01T05:00", "steps": 2, "diffusion_steps": 2, "output": d},
        })
        pred = np.load(run(cfg))
    assert pred.shape == (2, *shape), pred.shape
    valid = np.isfinite(pred)  # NaN only off Earth, so the same pixels in every step and channel
    assert (valid == valid[0, 0]).all() and valid[0, 0].mean() > 0.9, "unexpected NaNs in the forecast"
    print(f"forecast OK: {pred.shape[0]} steps of {pred.shape[1:]} saved")


if __name__ == "__main__":
    smoke_test()
