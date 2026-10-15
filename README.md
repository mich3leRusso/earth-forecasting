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

Hiding the GPU makes NATTEN, the attention library StormScope uses, install its pure-PyTorch
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
  (2–8 September), about 2,400 frames, 1.8 MB each, into `data/fci_ireland/`. Hours already on
  disk are skipped, so it can be stopped and restarted. Expect 10–30 s per frame.
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

### Publish: `configs/publish.yaml`

```bash
uv run hf auth login                             # once, with a token that can write
uv run earth configs/publish.yaml
```

Uploads `checkpoints/lora_1h/best.pt`, the config and a model card to the `hub.repo_id`
repository. Any config can then load it with `adapter=hf://<user>/<repo>`; it downloads on first
use.

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
configs/                 train.yaml, evaluate.yaml, publish.yaml
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
