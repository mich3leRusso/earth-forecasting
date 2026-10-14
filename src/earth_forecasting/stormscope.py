"""StormScope (Meteosat EU): load the model, crop it to a region, change its time step.

Run `python -m earth_forecasting.stormscope` for a smoke test on random input.
"""

import os

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

from earth2studio.models.px.stormscope_meteosat import StormScopeMeteosatEU
from earth_forecasting.config import region_of

# (lat range, lon range) in degrees
IRELAND = ((51.2, 55.6), (-11.0, -5.2))
N_INPUT = 6  # context frames per prediction; fixed by the checkpoint's input channels


def pixel_box(package, region, multiple: int = 32, patch: int = 4):
    """MTG pixel box (mtg_ylim, mtg_xlim) covering a lat/lon region.

    Sides are rounded up to `multiple` pixels, and the box starts on a `patch` boundary of
    the package grid so its tokens line up with the experts' positional embedding.
    """
    (lat0, lat1), (lon0, lon1) = region
    config = OmegaConf.load(package.resolve("config.yaml"))
    box_y = config.get("mtg_ylim", StormScopeMeteosatEU.Model_FCI_BBox[0])
    box_x = config.get("mtg_xlim", StormScopeMeteosatEU.Model_FCI_BBox[1])
    with xr.open_dataset(package.resolve("metadata.nc")) as metadata:
        lat, lon = metadata["lat"].values, metadata["lon"].values

    rows, cols = np.nonzero((lat >= lat0) & (lat <= lat1) & (lon >= lon0) & (lon <= lon1))
    if rows.size == 0:
        raise ValueError(f"Region {region} is outside the StormScope domain")

    def snap(i0, i1, n):
        size = -(-(i1 - i0) // multiple) * multiple
        start = min(max(i0 - (size - (i1 - i0)) // 2, 0), n - size)
        start -= start % patch  # stays in range: n and size are multiples of patch
        return start, start + size

    y0, y1 = snap(rows.min(), rows.max() + 1, lat.shape[0])
    x0, x1 = snap(cols.min(), cols.max() + 1, lat.shape[1])
    return (box_y[0] + y0, box_y[0] + y1), (box_x[0] + x0, box_x[0] + x1)


def fit_experts_to_crop(model: StormScopeMeteosatEU) -> None:
    """Make the DiT experts run on the crop.

    earth2studio crops the inputs, but each expert's positional embedding, token grid and
    detokenizer are sized for the full European grid. The full embedding is kept as is;
    each forward reads only the crop's window of it.
    """
    for expert in model.experts:
        dit = expert.model.model  # EDMPreconditioner -> ConcatConditionWrapper -> DiT
        ph, pw = dit.detokenizer.patch_size
        h, w = model.lat.shape[0] // ph, model.lat.shape[1] // pw
        r0, c0 = model._yi0 // ph, model._xi0 // pw
        tok, full_h, full_w = dit.tokenizer, dit._latent_h, dit._latent_w

        def tokenize(x, tok=tok, full_h=full_h, full_w=full_w, r0=r0, c0=c0, h=h, w=w):
            pos = tok.pos_embed.view(1, full_h, full_w, -1)[:, r0 : r0 + h, c0 : c0 + w]
            return tok.x_embedder(x).flatten(2).transpose(1, 2) + pos.reshape(1, h * w, -1)

        tok.forward = tokenize
        dit.attn_kwargs_forward["latent_hw"] = (h, w)
        dit.detokenizer.h_patches, dit.detokenizer.w_patches = h, w


def to_model_inputs(model: StormScopeMeteosatEU, frames: torch.Tensor) -> torch.Tensor:
    """The authors' input transform, as in create_generator: physical radiances -> raw counts ->
    normalized, off-Earth pixels set to 0. Missing pixels on Earth (NaN) also become 0;
    data.check_data counts them before this hides them."""
    return torch.nan_to_num(model.normalize(frames.to(model.invariants.device)))


def earth_mask(model: StormScopeMeteosatEU, device=None) -> torch.Tensor:
    """(H, W) boolean mask of the crop's pixels that see the Earth, on the model's device by default."""
    return torch.as_tensor(model.earth_mask, dtype=torch.bool, device=device or model.invariants.device)


def visualize_model(model, file_name="./model_architecture/model_graph.txt"):
    """Write every parameter name and whether it trains, one per line (handy for picking LoRA targets)."""
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        for name, param in model.named_parameters():
            f.write(f"{name} {param.requires_grad}\n")


def set_time_step(model: StormScopeMeteosatEU, step: np.timedelta64) -> None:
    """Use frames `step` apart: N_INPUT context frames up to now, one prediction `step` ahead."""
    model.input_times = np.arange(-N_INPUT + 1, 1) * step
    model.output_times = np.array([step])


def load_model(device: torch.device, region=None, compile: bool = True) -> StormScopeMeteosatEU:
    """Load StormScope; `region` = ((lat_min, lat_max), (lon_min, lon_max)) crops it, None = full Europe.

    Pass compile=False for training: natten's Flex backend refuses to be compiled for backprop.
    """
    package = StormScopeMeteosatEU.load_default_package()
    ylim, xlim = pixel_box(package, region) if region else (None, None)
    model = StormScopeMeteosatEU.load_model(package=package, mtg_ylim=ylim, mtg_xlim=xlim).to(device)
    model.eval()
    if region:
        fit_experts_to_crop(model)
    if compile:
        model.compile_model()
    return model


def model_from_config(cfg, device) -> StormScopeMeteosatEU:
    """StormScope cropped to cfg.data.region at a cfg.data.step_minutes step, uncompiled (training, evaluation)."""
    model = load_model(device, region=region_of(cfg), compile=False)
    set_time_step(model, np.timedelta64(cfg.data.step_minutes, "m"))
    return model


def smoke_test():
    """Load the Ireland crop and run each expert once on random input."""
    model = load_model(torch.device("cuda"), region=IRELAND, compile=False)
    H, W = model.lat.shape
    assert model._yi0 % 4 == 0 and model._xi0 % 4 == 0, "crop must start on a patch boundary"

    n_cond = 3 * (N_INPUT + 1) + len(model.variables) * N_INPUT + model.invariants.shape[1]
    x = torch.randn(1, len(model.variables), H, W, device="cuda")
    condition = torch.randn(1, n_cond, H, W, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, expert in enumerate(model.experts):
            out = expert(x, torch.ones(1, device="cuda"), condition=condition)
            assert out.shape == x.shape and torch.isfinite(out).all(), f"expert {i}: bad output"
    print(f"stormscope OK: crop {H}x{W} px, {len(model.experts)} experts run on it")


if __name__ == "__main__":
    smoke_test()
