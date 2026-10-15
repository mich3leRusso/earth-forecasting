"""Check downloaded frames before training: are they complete, and do they look like the model's inputs?

The `check` step (see cli.py) checks every period in the config and saves an example window as a
picture; training runs the same check before it starts.
Run `python -m earth_forecasting.checks` for a smoke test on synthetic frames (no network).
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from earth_forecasting.config import load_config
from earth_forecasting.data import SPLITS, FrameWindows, synthetic_frame_shape, write_synthetic_frames
from earth_forecasting.plots import window_figure
from earth_forecasting.stormscope import earth_mask, model_from_config


def check_data(model, datasets: dict, batch_size: int = 8, num_workers: int = 2, max_missing: float = 0.01) -> None:
    """Run every window through a DataLoader and the authors' input transform, and print what comes out.

    Per channel, on Earth pixels: mean and std of the normalized inputs (the authors normalized
    to ~0 / ~1 over Europe, so Ireland should be of that order), the share of missing pixels,
    the share outside the instrument's raw range, and how many frames have a constant channel.
    Raises if any channel misses more than `max_missing` of its pixels: training would silently
    see those as zeros. Out-of-range values and constant channels are only reported.
    """
    device = model.invariants.device
    on_earth = earth_mask(model)
    lo, hi = model.min_values.view(-1, 1) - 1e-3, model.max_values.view(-1, 1) + 1e-3  # (C, 1)
    problems = []
    for split, dataset in datasets.items():
        C = len(model.variables)
        n, missing, out, const = (torch.zeros(C, device=device) for _ in range(4))
        total, total_sq = torch.zeros(C, dtype=torch.float64, device=device), torch.zeros(C, dtype=torch.float64, device=device)
        gappy = []
        with torch.no_grad():
            for frames, times in DataLoader(dataset, batch_size, num_workers=num_workers):
                x = model.normalize(frames.to(device))[..., on_earth]  # (B, frames, C, pixels); NaN = missing
                nan = torch.isnan(x)
                v = torch.where(nan, 0.0, x)
                n += (~nan).sum((0, 1, 3))
                missing += nan.sum((0, 1, 3))
                total += v.sum((0, 1, 3)).double()
                total_sq += (v.double() ** 2).sum((0, 1, 3))
                out += ((x < lo) | (x > hi)).sum((0, 1, 3))
                const += (v.std(-1) == 0).sum((0, 1))
                gappy += [t for t, f in zip(times.tolist(), nan.float().mean((1, 2, 3)).tolist()) if f > max_missing]

        pixels = n + missing
        mean = total / n.double()
        std = (total_sq / n.double() - mean**2).clamp(min=0).sqrt()
        print(f"\n{split}: {len(dataset)} windows")
        print(f"{'channel':<10}{'mean':>8}{'std':>8}{'missing %':>11}{'out of range %':>16}{'constant frames':>17}")
        for i, name in enumerate(model.variables):
            print(f"{name:<10}{mean[i]:>8.2f}{std[i]:>8.2f}{100 * missing[i] / pixels[i]:>11.3f}"
                  f"{100 * out[i] / pixels[i]:>16.3f}{int(const[i]):>17}")
        worst = int((missing / pixels).argmax())
        if missing[worst] / pixels[worst] > max_missing:
            problems.append(f"{split}: {model.variables[worst]} misses {100 * missing[worst] / pixels[worst]:.1f}% of its pixels")
        if gappy:
            first = [str(np.datetime64(int(t), "s")) for t in gappy[:5]]
            print(f"{len(gappy)} windows miss more than {100 * max_missing:.0f}% of their pixels, e.g. anchors {first}")
    if problems:
        raise ValueError("data has gaps that training would fill with zeros:\n" + "\n".join(problems))


def run(cfg) -> None:
    """Step `check`: check the data of every period in the config and save an example window."""
    model = model_from_config(cfg, torch.device("cuda"))
    step = np.timedelta64(cfg.data.step_minutes, "m")
    datasets = {name: FrameWindows(*cfg.data[name], cfg.data.dir, step) for name in SPLITS if name in cfg.data}
    figure = Path(cfg.output.tensorboard).parent / "data_check.png"
    example = list(datasets.values())[-1]
    window_figure(model, *example[len(example) // 2]).savefig(figure, dpi=120, bbox_inches="tight")
    print(f"example window saved to {figure}")
    check_data(model, datasets)
    print("\ndata OK")


def smoke_test():
    """Clean synthetic frames pass the check; a window with an all-NaN frame is rejected."""
    cfg = load_config(args=[])
    shape = synthetic_frame_shape(cfg)
    model = model_from_config(cfg, torch.device("cuda"))
    with tempfile.TemporaryDirectory() as d:
        write_synthetic_frames(d, "2026-07-01T00", n_frames=7, shape=shape)  # one window, anchored at 05:00
        window = {"smoke": FrameWindows("2026-07-01T05:00", "2026-07-01T06:00", d)}
        check_data(model, window, num_workers=0)

        np.save(Path(d) / "2026-07-01T03:00.npy", np.full(shape, np.nan, dtype=np.float32))  # 1 of 7 frames missing
        try:
            check_data(model, window, num_workers=0)
        except ValueError:
            pass
        else:
            raise AssertionError("a window with an all-NaN frame should be rejected")
    print("checks OK: clean frames pass, missing pixels are caught")


if __name__ == "__main__":
    smoke_test()
