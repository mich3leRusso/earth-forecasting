# earth-forecasting

Forecasts Meteosat satellite imagery over Ireland one and two hours ahead. It fine-tunes NVIDIA's
[StormScope-Meteosat](https://huggingface.co/nvidia/stormscope-meteosat), a diffusion transformer
pre-trained on Europe at 10-minute steps, with LoRA adapters so it works on hourly frames over
Ireland. Runs on a laptop GPU with 8 GB of memory.

The trained adapter is on the Hugging Face Hub:
[`michele007rok/StormScope_Ireland`](https://huggingface.co/michele007rok/StormScope_Ireland).

## Install

You need Linux, an NVIDIA GPU (8 GB is enough), [uv](https://docs.astral.sh/uv/) and an account on
the [EUMETSAT Data Store](https://data.eumetsat.int) (free) for the satellite data.

```bash
sudo apt install build-essential                 # C compiler, used by Triton at runtime
git clone https://github.com/mich3leRusso/earth-forecasting
cd earth-forecasting
CUDA_VISIBLE_DEVICES="" uv sync                  # hide the GPU only while installing
```

Without uv, plain pip works too, with the same pinned versions:

```bash
CUDA_VISIBLE_DEVICES="" pip install -r requirements.txt && pip install --no-deps -e .
```

The notebook does this by itself when the project isn't installed (see below). Hiding the GPU makes NATTEN, the attention library StormScope uses, install its pure-PyTorch
backend instead of compiling CUDA kernels, which fails unless your CUDA toolkit matches PyTorch's
exactly. Everything uses the GPU normally afterwards.

### EUMETSAT credentials

The download needs the API key and secret of your EUMETSAT account, which you can generate at
[api.eumetsat.int/api-key](https://api.eumetsat.int/api-key/) once logged in. Create
`src/earth_forecasting/constants.py` with them:

```bash
cat > src/earth_forecasting/constants.py <<'EOF'
EUMETSAT_KEY = "your-consumer-key"
EUMETSAT_SECRET = "your-consumer-secret"
EOF
```

The file is git-ignored, so the keys never reach GitHub. Exporting `EUMETSAT_CONSUMER_KEY` and
`EUMETSAT_CONSUMER_SECRET` as environment variables works too.

The StormScope weights (2.4 GB) download by themselves on first use.

## Run

Everything goes through one command; the config says what to do:

```bash
uv run earth <config.yaml> [key=value ...]
```

Each config has a `steps` list, run in order, picked from `download`, `check`, `train`,
`evaluate`, `forecast` and `publish`. Any value can be overridden on the command line, e.g.
`steps=[train]` or `train.epochs=4`.

### Train: `configs/train.yaml`

```bash
uv run earth configs/train.yaml                  # download → train → evaluate
uv run tensorboard --logdir runs                 # in another terminal: curves at localhost:6006
```

- **download** fetches hourly frames for training (1 June – 31 August 2026) and validation
  (2–8 September): about 2,400 hours, so plan for several hours (see
  [How the download works](#how-the-download-works)).
- **train** checks the data, then trains LoRA adapters for 12 epochs (about 14 min each) and keeps
  the epoch with the lowest validation loss: `checkpoints/lora_1h/best.pt`.
- **evaluate** forecasts 20 validation times at +1h and +2h and scores them (see below).

### Evaluate: `configs/evaluate.yaml`

```bash
uv run earth configs/evaluate.yaml               # download → evaluate
```

Scores the adapter from the Hub on the test week (1–7 October 2026), which no model has seen. For
20 start times it forecasts +1h and +2h and compares three forecasts with what the satellite
observed:

- **fine-tuned:** StormScope with the adapter
- **base:** the same model with the adapter switched off
- **persistence:** the last observed frame, repeated

It prints the RMSE per channel and lead time and saves it to `runs/eval_test/rmse_test.csv`. Both
models get the same diffusion noise for each date, so their difference comes from the adapter
alone. To score another adapter, e.g. your own: `adapter=checkpoints/lora_1h/best.pt`.

### Forecast: `configs/forecast.yaml`

```bash
uv run earth configs/forecast.yaml                                         # from 12 October 2026, 12:00
uv run earth configs/forecast.yaml forecast.start=2026-10-14T09:00 forecast.steps=3
```

Makes one forecast with the adapter from the Hub: it fetches the six hours before
`forecast.start` (if they aren't on disk already), predicts `forecast.steps` hours ahead and saves
the result to `runs/forecasts/forecast_<start>.npy`, an array of shape (hours, 16 channels, 128,
224) in the satellite's physical units. `adapter=null` runs the untouched base model instead.

### Publish: `configs/publish.yaml`

```bash
uv run hf auth login                             # once, with a token that can write
uv run earth configs/publish.yaml
```

Uploads `checkpoints/lora_1h/best.pt`, the config and a model card to the `hub.repo_id`
repository. Any config can then load it with `adapter=hf://<user>/<repo>`; it downloads on first
use.

## How the download works

You don't run it separately: `download` is the first step of each config, and it fetches exactly
the hours that config needs (its training, validation and test periods, plus the few hours before
and after them that the samples use). Everything lands in one folder, `data/fci_ireland/`, as one
small file per hour named after its time, e.g. `2026-07-10T12:00.npy`. Training, validation and
test simply pick their files by date.

EUMETSAT doesn't let you download a region: every hour comes as the whole Earth disk, cut into 40
horizontal strips. The code downloads only the 2 strips that cover Ireland (about 45 MB instead of
1 GB), cuts out the Ireland box, saves it and deletes the rest. Each hour takes 10–30 seconds,
depending on how busy EUMETSAT is.

If it's interrupted, just run the same command again: files already on disk are skipped. A few
hours don't exist on EUMETSAT's side, or come incomplete; they are reported as `skipped` and left
out, and so are the training samples that would need them.

## Changing what it does

Everything is in the YAML files under `configs/`; the code doesn't need to change. The usual
edits:

| To change | Edit | Notes |
|---|---|---|
| training or validation dates | `data.train`, `data.val` in `train.yaml` | `[start, end)`; keep validation after training, with a day in between |
| the test week | `data.test` in `evaluate.yaml` | after every training and validation date |
| when a forecast starts, how far ahead | `forecast.start`, `forecast.steps` in `forecast.yaml` | UTC, on the hour |
| how far ahead to evaluate | `eval.lead_steps` | the download also fetches the extra hours |
| which adapter is used | `adapter` | a local `.pt` file, `hf://<user>/<repo>`, or `null` for the base model |
| how long / how hard to train | `train.epochs`, `train.lr`, `train.optimizer`, `lora.rank`, `lora.blocks` | see the comments in `train.yaml` |
| where data and results go | `data.dir`, `output.checkpoints`, `output.tensorboard` | |
| the region | `data.region` (degrees) | in every config, and retrain: an adapter only works on the region and step it was trained on |
| the time step | `data.step_minutes` | same as the region: retrain, and re-download |

For a one-off change you don't have to edit a file: pass it on the command line, e.g.
`uv run earth configs/train.yaml train.epochs=4 data.train=[2026-07-01,2026-08-01]`.

## Smoke tests

Quick self-checks on synthetic data, no network needed (all but `data`, `hub` and `plots` need the
GPU):

```bash
for m in data hub plots stormscope checks train evaluate forecast; do uv run python -m earth_forecasting.$m; done
```

## How it works

- **Data:** Level 1c images from the FCI imager on Meteosat-12, 16 channels (visible, near-infrared,
  water vapour, infrared), one per hour, cropped to Ireland: 128 × 224 pixels. EUMETSAT serves the
  full Earth disk in 40 horizontal strips; only the 2 strips covering Ireland are downloaded.
- **Sample:** six consecutive hourly frames in, the next hour out. +2h feeds the +1h prediction
  back in.
- **Model:** StormScope's two diffusion transformers (DiT, 16 blocks each, neighborhood attention).
  We run them on the Ireland crop only, reading the matching window of their Europe-wide positional
  embedding.
- **Training:** the 613M original weights stay frozen. LoRA adapters (rank 8, about 0.1% of the
  weights) sit on the attention of the last 8 blocks of both experts and on the output layer.
  Muon optimizer, warmup then cosine decay, batch 1 with 8-step gradient accumulation.
- **Periods:** NVIDIA trained the base model on November 2024 – May 2026, so all our data is new
  to it. Training, validation and test are separate in time.

## Layout

```
configs/                 train.yaml, evaluate.yaml, forecast.yaml, publish.yaml
src/earth_forecasting/
    cli.py               the `earth` command: runs the config's steps
    config.py            loads a config plus key=value overrides
    stormscope.py        loads the model, crops it to a region, sets the time step
    data.py              downloads frames and turns them into training windows
    checks.py            checks the data before training
    train.py             LoRA training
    evaluate.py          forecast scores against the base model and persistence
    forecast.py          one forecast from real frames, saved as .npy
    hub.py               publishes an adapter and loads it back from the Hub
    plots.py             figures and tables for the notebook
```

## Notes

- Run one download at a time: reading a product can briefly take several GB of memory.
- A few hours are missing or incomplete on EUMETSAT's side; their windows are skipped.
- The model only sees the crop, so weather coming from just outside it (mostly the Atlantic, to
  the west) can't enter the forecast.
- The noise levels used in training are drawn log-uniformly in each expert's range; NVIDIA hasn't
  published theirs.
- StormScope-Meteosat is under the OpenMDW-1.1 license; the imagery is EUMETSAT FCI Level 1c data.
