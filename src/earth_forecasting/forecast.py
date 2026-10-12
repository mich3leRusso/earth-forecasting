"""Forecast with StormScope: fetch the latest frames from EUMETSAT and roll the model forward.

`earth-forecasting` runs a 2-hour forecast over Ireland with the base model (10-minute steps).
Run `python -m earth_forecasting.forecast` for a smoke test on a synthetic initial condition.
"""

from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

from earth2studio.data import fetch_data
from earth2studio.data.meteosat_fci import MeteosatFCI
from earth2studio.models.px.stormscope_meteosat import StormScopeMeteosatEU
from earth2studio.utils.coords import map_coords
from earth_forecasting.data import set_eumetsat_credentials
from earth_forecasting.stormscope import IRELAND, load_model


def fetch_initial_condition(model: StormScopeMeteosatEU, start_time: np.datetime64, device, ensemble_size: int = 1):
    """Fetch the MTG-I1 FCI frames the model needs to start at `start_time`, on the model grid.

    Returns (x, coords) with x shaped (ensemble, time, lead_time, variable, y, x).
    """
    set_eumetsat_credentials()
    bbox_2km = (tuple(model.mtg_ylim), tuple(model.mtg_xlim))  # fetch only the model's area
    bboxes = {"2km": bbox_2km, "1km": tuple((2 * lo, 2 * hi) for lo, hi in bbox_2km)}
    fci = {res: MeteosatFCI(resolution=res, pixel_bbox=bbox) for res, bbox in bboxes.items()}

    in_coords = model.input_coords()
    x_res = {}
    for res in ["2km", "1km"]:
        available_vars = fci[res].available_variables()
        if res == "1km":
            # use the 2km version if a variable exists at both resolutions
            available_vars -= fci["2km"].available_variables()
        x_res[res] = fetch_data(
            fci[res],
            time=np.array([start_time]),
            variable=[v for v in in_coords["variable"] if v in available_vars],
            lead_time=in_coords["lead_time"],
            device=device,
        )

    # 2x downsample 1km data to the common 2km grid, then put it on the model grid
    x, coords = StormScopeMeteosatEU.combine_1km_2km_inputs(*x_res["1km"], *x_res["2km"])
    x, coords = map_coords(x, coords, in_coords)

    x = x.expand(ensemble_size, -1, -1, -1, -1, -1)
    coords["ensemble"] = np.arange(ensemble_size)
    coords.move_to_end("ensemble", last=False)
    return x, coords


def forecast(model: StormScopeMeteosatEU, x: torch.Tensor, coords, n_steps: int = 12):
    """Roll the model forward `n_steps` steps of its own time step.

    Returns a list of (x, coords) on CPU; item 0 is the initial condition.
    """
    frames = []
    with torch.no_grad():
        for step, (x_pred, coords_pred) in enumerate(tqdm(model.create_iterator(x, coords), total=n_steps + 1)):
            frames.append((x_pred.cpu(), coords_pred))
            if step == n_steps:
                break
    return frames


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, region=IRELAND)

    # Any time with MTG-I1 FCI data (UTC). StormScopeMeteosatEU was trained on
    # Nov 2024 - May 2026, so pick start times outside that period.
    start_time = np.datetime64("2026-09-30T12:00")
    print(f"Forecasting for {start_time} on {device} with {model.__class__.__name__}...")
    x, coords = fetch_initial_condition(model, start_time, device)

    frames = forecast(model, x, coords, n_steps=12)  # 2 hours of 10-minute steps
    print(f"{len(frames) - 1} forecast steps, frame shape {tuple(frames[-1][0].shape)}")


def smoke_test():
    """One forecast step on the Ireland crop from random frames, with a short diffusion sampler."""
    model = load_model(torch.device("cuda"), region=IRELAND, compile=False)
    model.num_diffusion_steps = 4  # the real sampler uses 48; enough to exercise the loop

    in_coords = model.input_coords()
    coords = OrderedDict(in_coords, batch=np.arange(1), time=np.array([np.datetime64("2026-07-01T12:00")]))
    shape = [len(v) for v in coords.values()]  # (batch, time, lead_time, variable, y, x)
    x = model.denormalize(torch.randn(shape, device="cuda"))  # random frames in physical units

    frames = forecast(model, x, coords, n_steps=1)
    pred = frames[-1][0]
    on_earth = torch.as_tensor(model.earth_mask, dtype=torch.bool).cpu()
    assert len(frames) == 2 and pred.shape == (*shape[:2], 1, *shape[3:]), pred.shape
    assert torch.isfinite(pred[..., on_earth]).all(), "non-finite values on Earth pixels"
    print(f"forecast OK: prediction {tuple(pred.shape)}")


if __name__ == "__main__":
    smoke_test()
