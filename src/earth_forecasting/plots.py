"""Figures and tables for looking at the data, the training and the forecasts (used by the notebook).

Run `python -m earth_forecasting.plots` for a smoke test of the tables (no GPU, no network).
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from physicsnemo.experimental.peft import set_adapter_enabled

from earth_forecasting.evaluate import rollout
from earth_forecasting.stormscope import N_INPUT, to_model_inputs

MODELS = ["fine-tuned", "base", "persistence"]


def window_figure(model, frames, t, channels=("fci06vis", "fci105ir"), step_hours=1):
    """One row per channel: the normalized context frames, then any target frames, north up."""
    x = to_model_inputs(model, frames).cpu()
    fig, axes = plt.subplots(len(channels), len(frames), figsize=(2.2 * len(frames), 1.7 * len(channels)), squeeze=False)
    for row, name in enumerate(channels):
        c = list(model.variables).index(name)
        for col in range(len(frames)):
            hours = (col - N_INPUT + 1) * step_hours
            axes[row, col].imshow(x[col, c], origin="lower", cmap="gray")  # row 0 is the southern edge
            axes[row, col].set_xticks([]), axes[row, col].set_yticks([])
            axes[row, col].set_title(f"{name}  {'now' if hours == 0 else f'{hours:+g}h'}", fontsize=8)
    fig.suptitle(f"{np.datetime64(int(t), 's')} UTC (normalized)", fontsize=10)
    fig.tight_layout()
    return fig


@torch.no_grad()
def forecast_figure(model, frames, t, lead_steps, channel="fci105ir", seed=0, step_hours=1):
    """Rows: lead times. Columns: persistence, base model, fine-tuned model, what the satellite saw.

    Both models get the same diffusion noise, so differences come from the adapter alone.
    """
    observed = to_model_inputs(model, frames[N_INPUT : N_INPUT + lead_steps]).cpu()
    persistence = to_model_inputs(model, frames[N_INPUT - 1]).cpu()
    preds = {}
    for name, enabled in (("base", False), ("fine-tuned", True)):
        set_adapter_enabled(model, enabled)
        torch.manual_seed(seed)
        preds[name] = rollout(model, frames[:N_INPUT], t, lead_steps).cpu()
    set_adapter_enabled(model, True)

    c = list(model.variables).index(channel)
    fig, axes = plt.subplots(lead_steps, 4, figsize=(11, 2.6 * lead_steps), squeeze=False)
    for k in range(lead_steps):
        truth = observed[k, c]
        panels = {"persistence": persistence[c], "base": preds["base"][k, c], "fine-tuned": preds["fine-tuned"][k, c], "observed": truth}
        lo, hi = float(truth.min()), float(truth.max())
        for col, (name, image) in enumerate(panels.items()):
            ax = axes[k, col]
            ax.imshow(image, origin="lower", cmap="gray", vmin=lo, vmax=hi)
            ax.set_xticks([]), ax.set_yticks([])
            rmse = "" if name == "observed" else f"  RMSE {float(((image - truth) ** 2).mean().sqrt()):.3f}"
            ax.set_title(f"{name}  +{(k + 1) * step_hours:g}h{rmse}", fontsize=9)
    fig.suptitle(f"{channel}, forecast from {np.datetime64(int(t), 's')} UTC (normalized)", fontsize=10)
    fig.tight_layout()
    return fig


def training_curves(log_dir) -> pd.DataFrame:
    """Per-epoch training and validation loss from a run's TensorBoard logs (epoch 0 = base model)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    events.Reload()
    tags = [tag for tag in ("train/loss", "val/loss") if tag in events.Tags()["scalars"]]
    return pd.DataFrame({tag: {e.step: e.value for e in events.Scalars(tag)} for tag in tags}).sort_index().rename_axis("epoch")


def rmse_summary(csv_path):
    """(mean RMSE per lead and model, with the gain over persistence; RMSE per channel) from evaluate's CSV."""
    table = pd.read_csv(csv_path)
    mean = table.groupby(["lead_hours", "model"])["rmse"].mean().unstack("model")[MODELS].round(3)
    for name in ("fine-tuned", "base"):
        mean[f"{name} vs persistence"] = (1 - mean[name] / mean["persistence"]).map("{:+.0%}".format)
    per_channel = table.pivot_table(index="channel", columns=["lead_hours", "model"], values="rmse", sort=False)
    return mean, per_channel.reindex(columns=MODELS, level="model")


def smoke_test():
    """The tables come out with the right shape and numbers, from a fake CSV and a fake TensorBoard log."""
    from torch.utils.tensorboard import SummaryWriter

    with tempfile.TemporaryDirectory() as d:
        rows = [(lead, ch, m, r) for lead in (1, 2) for ch in ("a", "b") for m, r in zip(MODELS, (0.4, 0.5, 0.8))]
        pd.DataFrame(rows, columns=["lead_hours", "channel", "model", "rmse"]).to_csv(f"{d}/rmse.csv", index=False)
        mean, per_channel = rmse_summary(f"{d}/rmse.csv")
        assert mean.loc[1, "fine-tuned vs persistence"] == "+50%" and per_channel.shape == (2, 6)

        with SummaryWriter(f"{d}/run") as w:
            for epoch, (tr, va) in enumerate([(0.9, 0.8), (0.5, 0.6)], start=1):
                w.add_scalar("train/loss", tr, epoch)
                w.add_scalar("val/loss", va, epoch)
        curves = training_curves(f"{d}/run")
        assert list(curves.index) == [1, 2] and abs(curves.loc[2, "val/loss"] - 0.6) < 1e-6
        w = SummaryWriter(f"{d}/run")  # a late epoch-0 point must still come first
        w.add_scalar("val/loss", 0.9, 0)
        w.close()
        assert list(training_curves(f"{d}/run").index) == [0, 1, 2]
    print("plots OK: RMSE summary and training curves")


if __name__ == "__main__":
    smoke_test()
