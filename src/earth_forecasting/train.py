"""LoRA fine-tuning of StormScope over a region at a new time step (default: Ireland, 1 hour).

The base model predicts 10 minutes ahead from 6 frames 10 minutes apart. Here the frames are
`data.step_minutes` apart, and only LoRA adapters on attention layers are trained: every
original weight stays frozen.

`earth-train configs/train.yaml [key=value ...]` trains; progress goes to TensorBoard.
Run `python -m earth_forecasting.train` for a smoke test on synthetic frames (no network).
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from physicsnemo.diffusion.metrics.losses import MSEDSMLoss
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.experimental.peft import (
    LoRAConfig,
    apply_lora,
    print_trainable_parameters,
    save_adapter,
    split_params_for_optimizer,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from earth_forecasting.data import FrameWindows, crop_grid, write_synthetic_frames
from earth_forecasting.stormscope import load_model, set_time_step

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "train.yaml"


def visualize_model(model, file_name="./model_architecture/model_graph.txt"):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for name, param in model.named_parameters():
            f.write(f"{name} {param.requires_grad}\n")


def region_of(cfg):
    return tuple(cfg.data.region.lat), tuple(cfg.data.region.lon)


def setup_model(cfg, device):
    """Frozen StormScope on the region at the config's step, with LoRA on the configured attention layers."""
    model = load_model(device, region=region_of(cfg), compile=False)
    set_time_step(model, np.timedelta64(cfg.data.step_minutes, "m"))
    blocks = "|".join(str(b) for b in cfg.lora.blocks)
    layers = "|".join(cfg.lora.layers)
    apply_lora(model, LoRAConfig(rank=cfg.lora.rank, target_pattern=rf"blocks\.({blocks})\.attention\.({layers})$"))
    print_trainable_parameters(model)
    return model


def condition(model, context, times):
    """Conditioning built like StormScopeMeteosatEU._forward: solar angles, context frames, invariants."""
    B, L, C, H, W = context.shape
    offsets = np.concatenate([model.input_times, model.output_times])
    zen_azi = torch.stack([model._azimuth_zenith(np.datetime64(int(t), "s") + offsets) for t in times])
    invariants = model.invariants[:, :, model._yi0 : model._yi1, model._xi0 : model._xi1].expand(B, -1, -1, -1)
    return torch.cat([zen_azi.view(B, 3 * (L + 1), H, W), context.reshape(B, C * L, H, W), invariants], dim=1)


def batch_loss(model, losses, frames, times, expert):
    """Denoising loss of one expert: noise the target frame at a sigma in its range, denoise, weighted MSE."""
    x = torch.nan_to_num(model.normalize(frames.to(model.invariants.device)))  # radiances -> model scale
    lo, hi = model.expert_sigma_min[expert], model.expert_sigma_max[expert]
    # ponytail: log-uniform sigma inside the expert's range; NVIDIA's training distribution isn't published
    sigma = lo * (hi / lo) ** torch.rand(x.shape[0], device=x.device)
    with torch.autocast(x.device.type, dtype=torch.bfloat16):
        return losses[expert](x[:, -1], t=sigma, condition=condition(model, x[:, :-1], times))


@torch.no_grad()
def validate(model, losses, dataset, max_samples):
    """Mean loss over both experts with fixed noise, so values are comparable between checks."""
    with torch.random.fork_rng(devices=[model.invariants.device]):
        torch.manual_seed(0)
        values = [
            batch_loss(model, losses, frames[None], [t], expert).item()
            for frames, t in (dataset[i] for i in range(min(len(dataset), max_samples)))
            for expert in range(len(model.experts))
        ]
    return sum(values) / len(values)


def train(cfg):
    """Batch 1 with gradient accumulation, alternating experts; stops at train.max_hours or train.max_steps."""
    model = setup_model(cfg, torch.device("cuda"))
    losses = [MSEDSMLoss(e, EDMNoiseScheduler(sigma_data=float(e.sigma_data))) for e in model.experts]

    step = np.timedelta64(cfg.data.step_minutes, "m")
    train_set = FrameWindows(*cfg.data.train, cfg.data.dir, step)
    val_set = FrameWindows(*cfg.data.val, cfg.data.dir, step)
    if not len(train_set) or not len(val_set):
        raise ValueError(f"no complete samples on disk: train {len(train_set)}, val {len(val_set)}. Run earth-download first.")
    loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=cfg.train.num_workers, drop_last=True)

    groups = split_params_for_optimizer(model)
    optimizer = torch.optim.AdamW(groups["lora"] + groups["extras"], lr=cfg.train.lr)
    ckpt_dir = Path(cfg.output.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.output.tensorboard)
    writer.add_text("config", f"```\n{OmegaConf.to_yaml(cfg)}\n```")

    seen, t0 = 0, time.time()
    deadline = t0 + cfg.train.max_hours * 3600
    done = lambda: time.time() > deadline or seen == cfg.train.max_steps

    # best.pt is saved only when validation beats the untouched base model
    best = validate(model, losses, val_set, cfg.train.val_samples)
    writer.add_scalar("val/loss", best, 0)
    print(f"train samples {len(train_set)}, val samples {len(val_set)}, val loss before training {best:.4f}")
    while not done():
        for frames, times in loader:
            expert = seen % len(model.experts)
            loss = batch_loss(model, losses, frames, times, expert)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at sample {seen} (anchor {times.tolist()})")
            (loss / cfg.train.grad_accum).backward()
            seen += 1
            writer.add_scalar(f"train/loss_expert{expert}", loss.item(), seen)
            if seen % cfg.train.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            if seen % cfg.train.val_every == 0:
                val = validate(model, losses, val_set, cfg.train.val_samples)
                writer.add_scalar("val/loss", val, seen)
                writer.add_scalar("perf/sec_per_sample", (time.time() - t0) / seen, seen)
                writer.add_scalar("perf/gpu_peak_gb", torch.cuda.max_memory_allocated() / 1e9, seen)
                print(f"sample {seen}  val {val:.4f}  {(time.time() - t0) / seen:.2f}s/sample")
                if val < best:
                    best = val
                    save_adapter(model, ckpt_dir / "best.pt")
            if done():
                break
    save_adapter(model, ckpt_dir / "last.pt")
    writer.close()
    return model


def main() -> None:
    """earth-train [config.yaml] [key=value ...]"""
    args = sys.argv[1:]
    path = args.pop(0) if args and "=" not in args[0] else DEFAULT_CONFIG
    train(OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(args)))


def smoke_test():
    """A few steps on 9 synthetic hourly frames: data -> loss -> LoRA update -> checkpoints and TensorBoard logs."""
    cfg = OmegaConf.load(DEFAULT_CONFIG)
    box, variables, _ = crop_grid(region_of(cfg))
    shape = (len(variables), box[0][1] - box[0][0], box[1][1] - box[1][0])
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        write_synthetic_frames(d, "2026-07-01T00", n_frames=9, shape=shape)  # full windows for anchors 05:00-07:00
        cfg = OmegaConf.merge(cfg, {
            "data": {"dir": d, "train": ["2026-07-01T05:00", "2026-07-01T07:00"], "val": ["2026-07-01T07:00", "2026-07-01T08:00"]},
            "train": {"max_steps": 4, "grad_accum": 2, "val_every": 4, "val_samples": 1, "num_workers": 0},
            "output": {"checkpoints": f"{d}/ckpt", "tensorboard": f"{d}/runs"},
        })
        model = train(cfg)

        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert trainable and all("lora" in n for n in trainable), "only LoRA parameters should train"
        assert any(p.abs().sum() > 0 for n, p in trainable.items() if "lora_B" in n), "lora_B never moved off zero"
        assert (Path(d) / "ckpt" / "last.pt").exists(), "no adapter saved"
        assert list(Path(d, "runs").glob("events.out.tfevents*")), "no TensorBoard log written"
    print(f"train OK: {time.time() - t0:.0f}s, peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    smoke_test()
