"""Configuration: every command reads one YAML file from configs/, plus optional key=value overrides."""

import sys
from pathlib import Path

from omegaconf import OmegaConf

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def load_config(default: str = "train.yaml", args=None):
    """Build the config from `[config.yaml] [key=value ...]` (the command line, or `args`).

    The path defaults to configs/<default>; overrides use dotted keys, e.g. `train.lr=1e-4`.
    """
    args = list(sys.argv[1:] if args is None else args)
    path = args.pop(0) if args and "=" not in args[0] else CONFIGS / default
    return OmegaConf.merge(OmegaConf.load(path), OmegaConf.from_dotlist(args))


def region_of(cfg):
    """The config's region as ((lat_min, lat_max), (lon_min, lon_max))."""
    return tuple(cfg.data.region.lat), tuple(cfg.data.region.lon)
