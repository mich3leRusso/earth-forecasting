import numpy as np
import torch

from .model.StormScope import IRELAND, forecast, load_data, load_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, region=IRELAND)

    # Any time with MTG-I1 FCI data (UTC). StormScopeMeteosatEU was trained on
    # Nov 2024 - May 2026, so pick start times outside that period.
    start_time = np.datetime64("2026-06-30T12:00")
    x, coords = load_data(model, start_time, device)

    frames = forecast(model, x, coords, n_steps=12)  # 2 hours of 10-minute steps
    print(f"{len(frames) - 1} forecast steps, frame shape {tuple(frames[-1][0].shape)}")
