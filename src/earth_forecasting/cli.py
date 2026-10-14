"""One command for everything: `uv run earth [config.yaml] [key=value ...]`.

The config's `steps` list says what to do, in order:
    download   fetch the frames of every period in `data` (frames already on disk are skipped)
    check      check those frames and save an example window
    train      LoRA fine-tuning (also checks the data first)
    evaluate   forecast the evaluation period and score it against the base model and persistence
    forecast   one forecast from `forecast.start`, saved to disk
    publish    upload the adapter to Hugging Face
Override it like any other key, e.g. `uv run earth configs/evaluate.yaml steps=[forecast]`.
The config defaults to configs/train.yaml.
"""

import sys

from earth_forecasting import checks, data, evaluate, forecast, hub, train
from earth_forecasting.config import load_config

STEPS = {
    "download": data.run,
    "check": checks.run,
    "train": train.run,
    "evaluate": evaluate.run,
    "forecast": forecast.run,
    "publish": hub.run,
}
# config sections each step reads, besides `data` and `output`
NEEDS = {"train": ["lora", "train"], "evaluate": ["eval"], "forecast": ["forecast"], "publish": ["lora", "hub"]}


def main() -> None:
    if sys.argv[1:2] in (["-h"], ["--help"]):
        print(__doc__)
        return
    cfg = load_config()
    steps = list(cfg.get("steps") or [])
    unknown = [s for s in steps if s not in STEPS]
    if not steps or unknown:
        raise SystemExit(f"steps must be a non-empty list of {list(STEPS)}; got {steps}")
    # fail now, not after hours of downloading, if a later step's section is missing
    missing = sorted({f"{s}: {sec}" for s in steps for sec in NEEDS.get(s, []) if sec not in cfg})
    if missing:
        raise SystemExit("the config lacks sections these steps need: " + ", ".join(missing))

    for step in steps:
        print(f"\n=== {step} ===")
        STEPS[step](cfg)


if __name__ == "__main__":
    main()
