"""LoRA fine-tuning of StormScope over a region at a new time step (default: Ireland, 1 hour).

The base model predicts 10 minutes ahead from 6 frames 10 minutes apart. Here the frames are
`data.step_minutes` apart, and only LoRA adapters on attention layers are trained: every
original weight stays frozen.

The `train` step (see cli.py) trains; progress goes to TensorBoard.
Run `python -m earth_forecasting.train` for a smoke test on synthetic frames (no network).
"""

import math
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

from earth_forecasting.checks import check_data
from earth_forecasting.config import load_config
from earth_forecasting.data import FrameWindows, synthetic_frame_shape, write_synthetic_frames
from earth_forecasting.stormscope import model_from_config, to_model_inputs


def setup_model(cfg, device):
    """Frozen StormScope on the region at the config's step, with LoRA on the configured attention layers."""
    model = model_from_config(cfg, device)
    blocks = "|".join(str(b) for b in cfg.lora.blocks)
    layers = "|".join(cfg.lora.layers)
    pattern = rf"blocks\.({blocks})\.attention\.({layers})$"
    if cfg.lora.output_layer:
        pattern += r"|detokenizer\.proj_layer\.output_projection$"
    apply_lora(model, LoRAConfig(rank=cfg.lora.rank, target_pattern=pattern))
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
    x = to_model_inputs(model, frames)
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


def make_optimizer(cfg, params, optimizer_steps):
    """Optimizer and learning-rate schedule from cfg.train: linear warmup, then cosine decay to min_lr * lr."""
    t = cfg.train
    if t.optimizer == "muon":
        # Muon orthogonalizes each 2-D update; match_rms_adamw rescales it to AdamW's size, so `lr` means the same
        optimizer = torch.optim.Muon(params, lr=t.lr, weight_decay=t.weight_decay, adjust_lr_fn="match_rms_adamw")
    elif t.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay)
    else:
        raise ValueError(f"train.optimizer must be muon or adamw, got {t.optimizer}")

    warmup = max(1, round(t.warmup * optimizer_steps))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(1, optimizer_steps - warmup), 1.0)
        return t.min_lr + (1 - t.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train(cfg):
    """Batch 1 with gradient accumulation, alternating experts, for train.epochs epochs.

    Validation, best.pt and the TensorBoard points come once per epoch. train.max_hours and
    train.max_steps (samples, for quick tests) can stop it earlier.
    """
    model = setup_model(cfg, torch.device("cuda"))
    losses = [MSEDSMLoss(e, EDMNoiseScheduler(sigma_data=float(e.sigma_data))) for e in model.experts]

    step = np.timedelta64(cfg.data.step_minutes, "m")
    train_set = FrameWindows(*cfg.data.train, cfg.data.dir, step)
    val_set = FrameWindows(*cfg.data.val, cfg.data.dir, step)
    if not len(train_set) or not len(val_set):
        raise ValueError(f"no complete samples on disk: train {len(train_set)}, val {len(val_set)}. Run the download step first.")
    check_data(model, {"train": train_set, "val": val_set}, num_workers=cfg.train.num_workers)
    loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=cfg.train.num_workers, drop_last=True)

    groups = split_params_for_optimizer(model)
    params = groups["lora"] + groups["extras"]
    total = cfg.train.max_steps or cfg.train.epochs * len(train_set)  # samples; sets the schedule length
    optimizer, schedule = make_optimizer(cfg, params, max(1, total // cfg.train.grad_accum))
    ckpt_dir = Path(cfg.output.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.output.tensorboard)
    writer.add_text("config", f"```\n{OmegaConf.to_yaml(cfg)}\n```")

    # epoch 0 is the untouched base model; best.pt is saved only when an epoch beats it
    best = validate(model, losses, val_set, cfg.train.val_samples)
    writer.add_scalar("val/loss", best, 0)
    print(f"{len(train_set)} training windows, {len(val_set)} validation windows, {cfg.train.optimizer}")
    print(f"epoch 0/{cfg.train.epochs}  val {best:.4f}  (base model)")

    seen, t0 = 0, time.time()
    deadline = t0 + cfg.train.max_hours * 3600
    for epoch in range(1, cfg.train.epochs + 1):
        epoch_start = time.time()
        expert_losses = [[] for _ in model.experts]
        grad_norms = []
        for frames, times in loader:
            expert = seen % len(model.experts)
            loss = batch_loss(model, losses, frames, times, expert)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in epoch {epoch} (anchor {times.tolist()})")
            (loss / cfg.train.grad_accum).backward()
            seen += 1
            expert_losses[expert].append(loss.item())
            if seen % cfg.train.grad_accum == 0:
                grad_norms.append(torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip).item())
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
            if seen >= total or time.time() > deadline:
                break

        val = validate(model, losses, val_set, cfg.train.val_samples)
        train_loss = sum(map(sum, expert_losses)) / max(1, sum(map(len, expert_losses)))
        for i, values in enumerate(expert_losses):
            if values:
                writer.add_scalar(f"train/loss_expert{i}", sum(values) / len(values), epoch)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val, epoch)
        writer.add_scalar("optim/lr", schedule.get_last_lr()[0], epoch)
        if grad_norms:
            writer.add_scalar("optim/grad_norm", sum(grad_norms) / len(grad_norms), epoch)
        writer.add_scalar("perf/minutes_per_epoch", (time.time() - epoch_start) / 60, epoch)
        writer.add_scalar("perf/gpu_peak_gb", torch.cuda.max_memory_allocated() / 1e9, epoch)
        print(f"epoch {epoch}/{cfg.train.epochs}  train {train_loss:.4f}  val {val:.4f}  {(time.time() - epoch_start) / 60:.1f} min")
        if val < best:
            best = val
            save_adapter(model, ckpt_dir / "best.pt")
        if seen >= total or time.time() > deadline:
            break
    save_adapter(model, ckpt_dir / "last.pt")
    writer.close()
    return model


def run(cfg) -> None:
    """Step `train`; the model is dropped on return, freeing the GPU for the steps after it."""
    train(cfg)
    torch.cuda.empty_cache()


def smoke_test():
    """A few steps on 9 synthetic hourly frames: data -> loss -> LoRA update -> checkpoints and TensorBoard logs."""
    cfg = load_config(args=[])
    # schedule over 100 steps: ramps up during warmup, peaks at lr, ends near min_lr * lr
    optimizer, schedule = make_optimizer(cfg, [torch.zeros(4, 4, requires_grad=True)], 100)
    lrs = []
    for _ in range(100):
        lrs.append(schedule.get_last_lr()[0])
        optimizer.step()
        schedule.step()
    warmup = round(cfg.train.warmup * 100)
    assert lrs[0] < lrs[warmup - 1] and abs(max(lrs) - cfg.train.lr) < 1e-9, "warmup should ramp up to lr"
    assert abs(lrs[-1] - cfg.train.min_lr * cfg.train.lr) < 0.01 * cfg.train.lr, "cosine should end near min_lr * lr"

    shape = synthetic_frame_shape(cfg)
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        write_synthetic_frames(d, "2026-07-01T00", n_frames=9, shape=shape)  # full windows for anchors 05:00-07:00
        cfg = OmegaConf.merge(cfg, {
            "data": {"dir": d, "train": ["2026-07-01T05:00", "2026-07-01T07:00"], "val": ["2026-07-01T07:00", "2026-07-01T08:00"]},
            "train": {"max_steps": 4, "grad_accum": 2, "val_samples": 1, "num_workers": 0},
            "lora": {"output_layer": True},
            "output": {"checkpoints": f"{d}/ckpt", "tensorboard": f"{d}/runs"},
        })
        model = train(cfg)

        trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert trainable and all("lora" in n for n in trainable), "only LoRA parameters should train"
        adapted = {n.rsplit(".", 1)[0] for n in trainable}  # modules holding lora_A / lora_B
        expected = len(model.experts) * (len(cfg.lora.blocks) * len(cfg.lora.layers) + 1)
        assert len(adapted) == expected, f"{len(adapted)} adapted layers, expected {expected}"
        assert sum(n.endswith("detokenizer.proj_layer.output_projection") for n in adapted) == len(model.experts)
        assert any(p.abs().sum() > 0 for n, p in trainable.items() if "lora_B" in n), "lora_B never moved off zero"
        assert (Path(d) / "ckpt" / "last.pt").exists(), "no adapter saved"
        assert list(Path(d, "runs").glob("events.out.tfevents*")), "no TensorBoard log written"
    print(f"train OK: {time.time() - t0:.0f}s, peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    smoke_test()
