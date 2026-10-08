"""StormScope (Meteosat EU) nowcasting: load the model, fetch FCI data, run the forecast."""

import numpy as np
import torch
import xarray as xr
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

from earth2studio.data import fetch_data
from earth2studio.data.meteosat_fci import MeteosatFCI
from earth2studio.models.px.stormscope_meteosat import StormScopeMeteosatEU
from earth2studio.utils.coords import map_coords

# (lat range, lon range) in degrees
IRELAND = ((51.2, 55.6), (-11.0, -5.2))


def _pixel_box(package, region, multiple: int = 32):
    """Convert a lat/lon region to the MTG pixel box (mtg_ylim, mtg_xlim) that covers it."""
    (lat0, lat1), (lon0, lon1) = region
    config = OmegaConf.load(package.resolve("config.yaml"))
    box_y = config.get("mtg_ylim", StormScopeMeteosatEU.Model_FCI_BBox[0])
    box_x = config.get("mtg_xlim", StormScopeMeteosatEU.Model_FCI_BBox[1])
    with xr.open_dataset(package.resolve("metadata.nc")) as metadata:
        lat, lon = metadata["lat"].values, metadata["lon"].values

    rows, cols = np.nonzero((lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1))
    if rows.size == 0:
        raise ValueError(f"Region {region} is outside the StormScope domain")

    # ponytail: size rounded up to a multiple of 32 so the network's patching divides it; unverified, lower if it errors
    def snap(i0, i1, n):
        size = -(-(i1 - i0) // multiple) * multiple
        start = min(max(i0 - (size - (i1 - i0)) // 2, 0), n - size)
        return start, start + size

    y0, y1 = snap(rows.min(), rows.max() + 1, lat.shape[0])
    x0, x1 = snap(cols.min(), cols.max() + 1, lat.shape[1])
    return (box_y[0] + y0, box_y[0] + y1), (box_x[0] + x0, box_x[0] + x1)


def load_model(device: torch.device, region=None) -> StormScopeMeteosatEU:
    """Load StormScope; `region` = ((lat_min, lat_max), (lon_min, lon_max)) crops it, None = full Europe."""
    package = StormScopeMeteosatEU.load_default_package()
    ylim, xlim = _pixel_box(package, region) if region else (None, None)
    model = StormScopeMeteosatEU.load_model(package=package, mtg_ylim=ylim, mtg_xlim=xlim).to(device)
    model.eval()
    model.compile_model()
    return model


def load_data(model: StormScopeMeteosatEU, start_time: np.datetime64, device: torch.device, ensemble_size: int = 1):
    """Fetch MTG-I1 FCI frames for `start_time` on the model grid.

    Returns (x, coords) with x shaped (ensemble, time, lead_time, variable, y, x).
    """
    load_dotenv()  # EUMETSAT credentials used by MeteosatFCI

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

    # 2x downsample 1km data to the common 2km grid
    x, coords = StormScopeMeteosatEU.combine_1km_2km_inputs(*x_res["1km"], *x_res["2km"])
    # put data on the model grid (reorders variables if needed)
    x, coords = map_coords(x, coords, in_coords)

    x = x.expand(ensemble_size, -1, -1, -1, -1, -1)
    coords["ensemble"] = np.arange(ensemble_size)
    coords.move_to_end("ensemble", last=False)
    return x, coords


def forecast(model: StormScopeMeteosatEU, x: torch.Tensor, coords, n_steps: int = 12):
    """Roll the model forward `n_steps` 10-minute steps.

    Returns a list of (x, coords) on CPU; item 0 is the initial condition.
    """
    frames = []
    with torch.no_grad():
        for step, (x_pred, coords_pred) in enumerate(tqdm(model.create_iterator(x, coords), total=n_steps + 1)):
            frames.append((x_pred.cpu(), coords_pred))
            if step == n_steps:
                break
    return frames
