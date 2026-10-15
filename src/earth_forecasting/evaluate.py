"""Forecast skill: the fine-tuned model against the base model and persistence.

The period is `data.test` if the config has one, otherwise `data.val`. At times spread evenly
over it, each model forecasts `eval.lead_steps` steps ahead from the six frames before (one
step at a time, feeding its own prediction back in), and every lead is scored against what the
satellite actually saw: RMSE of the normalized channels on Earth pixels, so all channels share
one scale. "base" is the same model with the LoRA adapters
switched off; "persistence" just repeats the last observed frame, the bar any forecast must clear.

The `evaluate` step (see cli.py) scores the config's adapter (see find_adapter), e.g.
`uv run earth configs/evaluate.yaml adapter=checkpoints/lora_1h/last.pt`.
Run `python -m earth_forecasting.evaluate` for a smoke test on synthetic frames (no network).
"""

import csv
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from physicsnemo.experimental.peft import LoRAConfig, apply_lora, load_adapter, save_adapter, set_adapter_enabled
from torch.utils.tensorboard import SummaryWriter

from earth_forecasting.config import load_config
from earth_forecasting.data import FrameWindows, synthetic_frame_shape, write_synthetic_frames
from earth_forecasting.hub import fetch_adapter
from earth_forecasting.stormscope import N_INPUT, earth_mask, model_from_config, to_model_inputs


def find_adapter(cfg):
    """The config's `adapter` if set (a path, or hf://<user>/<repo>[/<file>], downloaded on first use),
    else best.pt (or last.pt) in output.checkpoints, else None."""
    if cfg.get("adapter"):
        return fetch_adapter(cfg.adapter)
    checkpoints = cfg.get("output", {}).get("checkpoints")
    if not checkpoints:
        return None
    return next((p for p in (Path(checkpoints) / "best.pt", Path(checkpoints) / "last.pt") if p.is_file()), None)


def finetuned_model(cfg, device=None):
    """(model, adapter path): StormScope for the config with its adapter applied, or the base model if none."""
    model = model_from_config(cfg, device or torch.device("cuda"))
    adapter = find_adapter(cfg)
    if adapter:
        load_adapter(model, adapter)
    return model, adapter


def rollout(model, context, t, lead_steps):
    """Forecast `lead_steps` steps from N_INPUT physical frames ending at time `t` (seconds).

    Returns normalized predictions (lead_steps, C, H, W), the model's own scale.
    """
    coords = OrderedDict(model.input_coords(), batch=np.arange(1), time=np.array([np.datetime64(int(t), "s")]))
    x = context[None, None].to(model.invariants.device)  # (batch, time, lead_time, C, H, W)
    preds = []
    for k, (y, _) in enumerate(model.create_iterator(x, coords)):
        if k:  # item 0 is the initial condition
            preds.append(to_model_inputs(model, y[0, 0, 0]))
        if k == lead_steps:
            break
    return torch.stack(preds)


@torch.no_grad()
def evaluate(cfg, adapter, writer=None):
    """Print and return RMSE per lead and channel for the fine-tuned model, the base model and persistence."""
    model = model_from_config(cfg, torch.device("cuda"))
    load_adapter(model, adapter)
    if cfg.eval.diffusion_steps:
        model.num_diffusion_steps = cfg.eval.diffusion_steps
    step = np.timedelta64(cfg.data.step_minutes, "m")
    lead = cfg.eval.lead_steps
    period = "test" if "test" in cfg.data else "val"
    dataset = FrameWindows(*cfg.data[period], cfg.data.dir, step, lead_steps=lead)
    if not len(dataset):
        print(f"evaluation skipped: no {period} window has {lead} target frames on disk")
        return None
    picks = np.unique(np.linspace(0, len(dataset) - 1, min(cfg.eval.samples, len(dataset))).round().astype(int))

    on_earth = earth_mask(model)
    names = ("fine-tuned", "base", "persistence")
    squared = {n: torch.zeros(lead, len(model.variables), device=on_earth.device) for n in names}
    used = skipped = 0
    with torch.random.fork_rng(devices=[on_earth.device]):
        for i in picks:
            frames, t = dataset[i]
            if torch.isnan(frames[..., on_earth.cpu()]).any():  # the sampler can't start from missing pixels
                skipped += 1
                continue
            truth = to_model_inputs(model, frames[N_INPUT:])
            last = to_model_inputs(model, frames[N_INPUT - 1])
            squared["persistence"] += ((last - truth) ** 2)[..., on_earth].mean(-1)
            for name, enabled in (("fine-tuned", True), ("base", False)):
                set_adapter_enabled(model, enabled)
                torch.manual_seed(int(i))  # both models get the same diffusion noise, so only the adapter differs
                squared[name] += ((rollout(model, frames[:N_INPUT], t, lead) - truth) ** 2)[..., on_earth].mean(-1)
            used += 1
    set_adapter_enabled(model, True)
    if not used:
        print(f"evaluation skipped: all {skipped} picked windows have missing pixels")
        return None
    rmse = {n: (squared[n] / used).sqrt().cpu() for n in names}

    print(f"\nforecast RMSE (normalized units) over {used} {period} times, {skipped} skipped for missing pixels")
    for k in range(lead):
        hours = (k + 1) * cfg.data.step_minutes / 60
        print(f"\n+{hours:g}h{'':<6}" + "".join(f"{n:>13}" for n in names))
        for c, channel in enumerate(model.variables):
            print(f"{channel:<10}" + "".join(f"{rmse[n][k, c]:>13.3f}" for n in names))
        print(f"{'mean':<10}" + "".join(f"{rmse[n][k].mean():>13.3f}" for n in names))
        if writer:
            for n in names:
                writer.add_scalar(f"eval/rmse_{hours:g}h/{n}", rmse[n][k].mean().item(), 0)

    table = Path(cfg.output.tensorboard) / f"rmse_{period}.csv"  # one row per lead, channel and model
    table.parent.mkdir(parents=True, exist_ok=True)
    with open(table, "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["lead_hours", "channel", "model", "rmse"])
        for k in range(lead):
            for c, channel in enumerate(model.variables):
                rows.writerows([(k + 1) * cfg.data.step_minutes / 60, channel, n, f"{rmse[n][k, c]:.5f}"] for n in names)
    print(f"\ntable saved to {table}")
    return rmse


def run(cfg) -> None:
    """Step `evaluate`: score the config's adapter (see find_adapter) and log the scores to TensorBoard."""
    adapter = find_adapter(cfg)
    if adapter is None:
        raise FileNotFoundError("no adapter to score: set `adapter` in the config or train one first")
    with SummaryWriter(cfg.output.tensorboard) as writer:
        evaluate(cfg, adapter, writer)


def smoke_test():
    """Score a fresh adapter on 8 synthetic frames with a 2-step sampler: all three models get finite RMSEs."""
    cfg = load_config(args=[])
    shape = synthetic_frame_shape(cfg)
    with tempfile.TemporaryDirectory() as d:
        write_synthetic_frames(d, "2026-07-01T00", n_frames=8, shape=shape)  # anchor 05:00 has +1h and +2h
        model = model_from_config(cfg, torch.device("cuda"))
        apply_lora(model, LoRAConfig(rank=cfg.lora.rank, target_pattern=r"blocks\.15\.attention\.qkv$"))
        save_adapter(model, f"{d}/adapter.pt")
        del model
        cfg = OmegaConf.merge(cfg, {
            "data": {"dir": d, "test": ["2026-07-01T05:00", "2026-07-01T06:00"]},  # test takes precedence over val
            "eval": {"samples": 1, "diffusion_steps": 2},
            "output": {"tensorboard": d},
        })
        rmse = evaluate(cfg, f"{d}/adapter.pt")
        table = (Path(d) / "rmse_test.csv").read_text().splitlines()
        assert len(table) == 1 + 2 * shape[0] * 3, "CSV should hold every lead x channel x model"
    assert rmse is not None and all(torch.isfinite(r).all() and r.shape == (2, shape[0]) for r in rmse.values())
    # a fresh adapter is a no-op (lora_B = 0), so with shared noise both models must score the same
    assert torch.allclose(rmse["fine-tuned"], rmse["base"], atol=1e-3), "fine-tuned and base should match"
    print("evaluate OK: fine-tuned, base and persistence scored at +1h and +2h")


if __name__ == "__main__":
    smoke_test()
