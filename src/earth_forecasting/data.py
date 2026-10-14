"""MTG-FCI frames over a region: download once to .npy, then serve training samples.

An EUMETSAT product is the full Earth disk (~1 GB) cut into 40 horizontal chunks; only the
chunks that contain the crop are downloaded (~45 MB for Ireland), and deleted once cropped. Frames are stored as physical radiances on the model's
2 km grid, in the model's channel order; the model's own `normalize` turns them into
network inputs.

The `download` step (see cli.py) fetches every period the config defines (train, val, test).
Run `python -m earth_forecasting.data` for a smoke test on synthetic frames (no network).
"""

import bisect
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from earth2studio.data.meteosat_fci import MeteosatFCI, _sort_body_key
from earth2studio.models.px.stormscope_meteosat import StormScopeMeteosatEU
from earth_forecasting.config import region_of
from earth_forecasting.stormscope import N_INPUT, pixel_box

SPLITS = ("train", "val", "test")  # periods a config may define under `data`; each one is optional


def set_eumetsat_credentials() -> None:
    """MeteosatFCI reads its credentials from env vars; fill them from constants.py if it exists."""
    try:
        from earth_forecasting.constants import EUMETSAT_KEY, EUMETSAT_SECRET
    except ImportError:
        return  # rely on EUMETSAT_CONSUMER_KEY / EUMETSAT_CONSUMER_SECRET being set
    os.environ.setdefault("EUMETSAT_CONSUMER_KEY", EUMETSAT_KEY)
    os.environ.setdefault("EUMETSAT_CONSUMER_SECRET", EUMETSAT_SECRET)


def crop_grid(region):
    """Pixel box of the crop, the model's channel order, and the crop's row coordinates."""
    package = StormScopeMeteosatEU.load_default_package()
    box = pixel_box(package, region)
    origin_y = OmegaConf.load(package.resolve("config.yaml"))["mtg_ylim"][0]
    with xr.open_dataset(package.resolve("metadata.nc")) as metadata:
        variables = [str(v) for v in metadata["variable_state"].values]
        mtg_y = metadata["mtg_y"].values[box[0][0] - origin_y : box[0][1] - origin_y]
    return box, variables, mtg_y


FDHSI_CHUNKS = 40  # horizontal chunks per full-disk product, numbered 1..40 from the south
FDHSI_ROWS_2KM = 5568  # rows of the full disk at 2 km


def chunks_for_rows(i0: int, i1: int, rows: int = FDHSI_ROWS_2KM) -> set[int]:
    """Chunk numbers covering full-disk rows [i0, i1), counted from the south on a `rows`-row grid."""
    starts = [int(c * rows / FDHSI_CHUNKS) for c in range(FDHSI_CHUNKS)]  # first row of chunks 1..40
    return set(range(bisect.bisect_right(starts, i0), bisect.bisect_right(starts, i1 - 1) + 1))


class CroppedFCI(MeteosatFCI):
    """MeteosatFCI that downloads only the chunks covering `chunks`, not the whole ~1 GB product.

    earth2studio's reader counts rows from the first chunk it finds, so the crop is shifted by the
    first downloaded chunk's own start row (`start_position_row`, stored in every chunk file).
    """

    def __init__(self, *args, chunks: set[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.chunks = chunks

    def _download_product(self, product, product_cache: str) -> None:
        wanted = [n for n in product.entries if "BODY" in n and n.endswith(".nc") and _sort_body_key(n) in self.chunks]
        if len(wanted) != len(self.chunks):
            raise FileNotFoundError(f"product has chunks {sorted(map(_sort_body_key, wanted))}, need {sorted(self.chunks)}")
        for name in wanted:
            target = os.path.join(product_cache, os.path.basename(name))
            with product.open(entry=name) as stream, open(target + ".part", "wb") as f:
                shutil.copyfileobj(stream, f)
            os.replace(target + ".part", target)  # a half-written chunk never looks complete

    def _read_channel(self, product_dir, channel_name, pixel_roi=None):
        if pixel_roi is None:
            raise ValueError("CroppedFCI needs a pixel box")
        first = min(Path(product_dir).rglob("*BODY*.nc"), key=lambda p: _sort_body_key(p.name))
        start = int(self._ensure_dataset(str(first))[f"/data/{channel_name}/measured"]["start_position_row"][...]) - 1
        (i0, i1), cols = pixel_roi
        return super()._read_channel(product_dir, channel_name, ((i0 - start, i1 - start), cols))


def download_frames(start, end, out_dir: Path, region, step=np.timedelta64(1, "h")) -> None:
    """Save one .npy per `step` in [start, end) as (channels, y, x) float32. Re-running skips saved frames."""
    set_eumetsat_credentials()
    box, variables, mtg_y = crop_grid(region)
    chunks = chunks_for_rows(*box[0])  # chunks are bands of the disk, the same for both resolutions
    src = {
        "2km": CroppedFCI("2km", pixel_bbox=box, verbose=False, chunks=chunks),
        "1km": CroppedFCI("1km", pixel_bbox=tuple((2 * lo, 2 * hi) for lo, hi in box), verbose=False, chunks=chunks),
    }
    # a channel is taken at 1 km only when it doesn't exist at 2 km
    v2 = [v for v in variables if v in src["2km"].available_variables()]
    v1 = [v for v in variables if v not in v2]
    order = [(v1 + v2).index(v) for v in variables]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in tqdm(np.arange(np.datetime64(start, "m"), np.datetime64(end, "m"), step)):
        path = out_dir / f"{t}.npy"
        if path.exists():
            continue
        when = t.astype("datetime64[s]").astype(datetime)
        try:
            a2 = src["2km"](when, v2)
            if not np.allclose(a2.y.values, mtg_y):
                raise RuntimeError("downloaded rows are not in the model grid's order")
            a1 = src["1km"](when, v1).values[0]
            c, h, w = a1.shape
            a1 = a1.reshape(c, h // 2, 2, w // 2, 2).mean(axis=(2, 4))  # 1 km -> 2 km
            np.save(path, np.concatenate([a1, a2.values[0]])[order].astype(np.float32))
        except RuntimeError:
            raise
        except Exception as e:  # missing product or network error: skip this frame, a re-run retries it
            print(f"{t}: skipped ({type(e).__name__}: {e})")
        finally:
            shutil.rmtree(src["2km"].cache, ignore_errors=True)  # both resolutions share this cache


class FrameWindows(Dataset):
    """Samples anchored at each time t in [start, end) whose frames t-5*step ... t+lead_steps*step are all on disk.

    Item: (frames (N_INPUT + lead_steps, C, y, x) physical radiances, t in seconds since epoch).
    The first N_INPUT frames are the context, ending at the anchor; the rest are the targets.
    """

    def __init__(self, start, end, data_dir: Path, step=np.timedelta64(1, "h"), lead_steps: int = 1):
        self.data_dir = Path(data_dir)
        have = {np.datetime64(p.stem, "m") for p in self.data_dir.glob("*.npy")}
        self.offsets = np.arange(-N_INPUT + 1, lead_steps + 1) * step
        self.times = [
            t
            for t in np.arange(np.datetime64(start, "m"), np.datetime64(end, "m"), step)
            if all(t + o in have for o in self.offsets)
        ]

    def __len__(self) -> int:
        return len(self.times)

    def __getitem__(self, i):
        t = self.times[i]
        frames = np.stack([np.load(self.data_dir / f"{t + o}.npy") for o in self.offsets])
        return torch.from_numpy(frames), t.astype("datetime64[s]").astype(np.int64)


def run(cfg) -> None:
    """Step `download`: fetch every frame the config's periods need (frames already on disk are skipped)."""
    step = np.timedelta64(cfg.data.step_minutes, "m")
    lead = cfg.eval.lead_steps if "eval" in cfg else 1  # evaluation also needs the frames up to its last lead
    for name in (n for n in SPLITS if n in cfg.data):
        start, end = cfg.data[name]
        print(f"{name}: {start} -> {end}")
        # a sample anchored at t needs frames from t - (N_INPUT-1) steps to t + lead steps
        download_frames(
            np.datetime64(start, "m") - (N_INPUT - 1) * step,
            np.datetime64(end, "m") + lead * step,
            cfg.data.dir,
            region_of(cfg),
            step,
        )


def synthetic_frame_shape(cfg):
    """(channels, y, x) of the config's crop, for synthetic frames that match the real ones."""
    box, variables, _ = crop_grid(region_of(cfg))
    return len(variables), box[0][1] - box[0][0], box[1][1] - box[1][0]


def write_synthetic_frames(data_dir, start, n_frames, shape=(16, 128, 224), step=np.timedelta64(1, "h"), skip=()):
    """Random frames named like real ones; indices in `skip` are left out to simulate missing hours."""
    rng = np.random.default_rng(0)
    t0 = np.datetime64(start, "m")
    for i in range(n_frames):
        if i not in skip:
            np.save(Path(data_dir) / f"{t0 + i * step}.npy", rng.random(shape, dtype=np.float32))


def smoke_test():
    """Windows are built only where all 7 frames exist, in time order, with the right shapes."""
    with tempfile.TemporaryDirectory() as d:
        write_synthetic_frames(d, "2026-07-01T00", n_frames=12, skip={9})  # 00:00..11:00 without 09:00
        ds = FrameWindows("2026-07-01T00", "2026-07-02T00", d)
        # anchors need t-5h..t+1h: 05:00..07:00 work; 08:00-10:00 touch the missing 09:00; 11:00 has no t+1h
        expected = [np.datetime64(f"2026-07-01T0{h}:00") for h in (5, 6, 7)]
        assert ds.times == expected, ds.times
        frames, t = ds[0]
        assert frames.shape == (N_INPUT + 1, 16, 128, 224) and frames.dtype == torch.float32
        assert np.array_equal(frames[-1].numpy(), np.load(Path(d) / "2026-07-01T06:00.npy")), "target = anchor + 1h"
        assert t == np.datetime64("2026-07-01T05:00", "s").astype(np.int64)
    # chunk 37 of a real product starts at 2 km row 5011 and 1 km row 10022 (start_position_row - 1)
    assert chunks_for_rows(5011, 5150) == {37} and chunks_for_rows(5010, 5011) == {36}
    assert chunks_for_rows(5076, 5204) == {37, 38} and chunks_for_rows(10022, 10300, rows=2 * FDHSI_ROWS_2KM) == {37}
    print(f"data OK: {len(ds)} windows from 11 of 12 synthetic frames")


if __name__ == "__main__":
    smoke_test()
