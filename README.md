# earth-forecasting

This project forecasts satellite imagery over Ireland one hour ahead. It starts from NVIDIA's
[StormScope-Meteosat](https://huggingface.co/nvidia/stormscope-meteosat), a diffusion model that
predicts the next Meteosat image 10 minutes into the future, and fine-tunes it with LoRA so that it
works on hourly frames instead.

Everything here is built to run on a laptop with an 8 GB GPU. That constraint shaped most of the
decisions below, so it's worth reading the "How it works" section before changing anything.

## How it works

StormScope looks at the last six images from the Flexible Combined Imager (FCI) on Meteosat Third
Generation, sixteen spectral channels each, and generates the next one. Under the hood it is a
diffusion transformer split into two "experts": one handles the noisy early steps of sampling,
where the large-scale structure gets decided, and the other the fine details at the end.

The full model covers Europe on a 1120 × 2432 grid. That's far too big to train on 8 GB, so we
crop it to Ireland: 128 × 224 pixels at about 2 km, roughly 95 times fewer tokens. The crop needs a
small fix that earth2studio doesn't do on its own. Each expert has a positional embedding sized
for the whole of Europe, so for every forward pass we read just the window that matches Ireland.
The weights themselves are untouched.

We then change the time step from 10 minutes to one hour: six hourly frames go in, the frame one
hour later comes out. The original weights stay frozen. The only thing we train is a set of LoRA
adapters (rank 8) on the attention projections of the last 8 of 16 blocks in both experts. That's
about 590,000 parameters, 0.1% of the model.

Training only uses data from June 2026 onwards, because NVIDIA trained the base model on November
2024 to May 2026.

## Setup

You need an NVIDIA GPU, [uv](https://docs.astral.sh/uv/), and a C compiler (`sudo apt install
build-essential`); Triton uses the compiler to build its GPU kernels at runtime.

```bash
CUDA_VISIBLE_DEVICES="" uv sync
```

Hiding the GPU during the install isn't a typo. StormScope's attention comes from
[NATTEN](https://github.com/SHI-Labs/NATTEN), which ships as source code. If it sees a GPU while
installing, it tries to compile CUDA kernels, and that fails unless your CUDA toolkit matches
PyTorch's exactly. Without a GPU in sight it installs in seconds and falls back to its
PyTorch-only backend (Flex Attention). That backend is slower but gives the same results, and it
supports training. Once installed, everything sees the GPU normally.

To download data you also need an account on the
[EUMETSAT Data Store](https://data.eumetsat.int). Either export your API credentials:

```bash
export EUMETSAT_CONSUMER_KEY=...
export EUMETSAT_CONSUMER_SECRET=...
```

or put them in `src/earth_forecasting/constants.py` as `EUMETSAT_KEY` and `EUMETSAT_SECRET`. That
file is git-ignored, so the keys stay on your machine.

The model weights (about 2.4 GB) download from Hugging Face the first time you load the model, into
`~/.cache/earth2studio`.

## Downloading data

All settings live in [`configs/train.yaml`](configs/train.yaml): the region, the time step, the
training and validation periods, the LoRA settings and the training budget. The same file drives
the download, so you edit dates in one place:

```bash
uv run earth-download configs/train.yaml
```

Plan for the download time. EUMETSAT only serves full-disk products, about 1 GB each, so every
hour of data means downloading a whole gigabyte, cropping out Ireland and throwing the rest away.
On our connection that takes around 45 seconds per hour of data, and each saved frame takes only
1.8 MB of disk. The default config (four weeks of training data plus one week of validation) is
roughly 850 frames, which is an overnight job.

You can stop the download at any time. Frames that are already saved get skipped on the next run,
and an hour that EUMETSAT can't serve is just reported and skipped.

## Training

```bash
uv run earth-train configs/train.yaml
```

Any config value can be overridden from the command line without editing the file:

```bash
uv run earth-train configs/train.yaml train.max_hours=2 lora.rank=16
```

The batch size is 1, because that's what fits in memory, and gradients are accumulated over 8
samples before each optimizer step. Each sample trains one expert, alternating between the two, at
a noise level drawn from that expert's range. On an RTX 5060 Laptop a sample takes about 1.6
seconds and the run peaks around 4 GB of GPU memory. Training stops at `train.max_hours` (20 by
default).

Validation runs every `train.val_every` samples with fixed noise, so the numbers are comparable from
one check to the next. The very first validation happens before any training, which gives you the
base model's score as a baseline. `best.pt` is only written when the adapter beats that baseline;
`last.pt` is always written at the end. Both go to `checkpoints/lora_1h/`.

To watch progress:

```bash
uv run tensorboard --logdir runs
```

You'll see the training loss for each expert, the validation loss, seconds per sample and peak GPU
memory. The config the run used is saved under the Text tab.

## Publishing to Hugging Face

When you have an adapter you're happy with, set `hub.repo_id` in the config (for example
`your-name/stormscope-ireland-lora`), log in once, and publish:

```bash
uv run hf auth login
uv run earth-publish configs/train.yaml
```

This uploads the adapter (`best.pt`, or `last.pt` if validation never improved), the exact config it
was trained with, and a model card describing the setup. The repository is created as private
unless you set `hub.private: false`.

## Forecasting

```bash
uv run earth-forecasting
```

This fetches the latest frames for a fixed start time and runs a two-hour forecast over Ireland with
the base model in 10-minute steps. It doesn't load a fine-tuned adapter yet.

## Smoke tests

Each main module has a quick self-check you can run on its own. They use synthetic data, so they
don't need EUMETSAT credentials or a network connection (except for the first download of the
model weights):

```bash
uv run python -m earth_forecasting.data        # building training windows from frames on disk
uv run python -m earth_forecasting.stormscope  # loading the model and running it on the crop
uv run python -m earth_forecasting.train       # a few training steps, checkpoints, TensorBoard logs
uv run python -m earth_forecasting.forecast    # one forecast step from a random start
uv run python -m earth_forecasting.hub         # preparing the upload folder, without uploading
```

All of them except `data` and `hub` need the GPU.

## Project layout

```
configs/train.yaml          every setting for downloading and training
src/earth_forecasting/
    stormscope.py           load the model, crop it to a region, change its time step
    data.py                 download frames and turn them into training samples
    train.py                LoRA fine-tuning
    forecast.py             forecasts from live EUMETSAT data
    hub.py                  publish an adapter to Hugging Face
    experimental/           Mamba and hybrid Mamba/attention blocks (not used by the pipeline)
```

## Things to keep in mind

- The noise levels used in training are drawn uniformly on a log scale inside each expert's range.
  NVIDIA hasn't published the distribution they used, so this is an educated guess and a natural
  thing to tune.
- The model isn't compiled during training, because NATTEN's Flex backend doesn't support it for
  backpropagation. Compiling is still used for plain forecasting.
- Moving from 10-minute to hourly steps is a big change for the model, and LoRA on half the
  attention layers may not be enough to fully adapt to it. The validation curve will tell you.
- StormScope-Meteosat is released under the OpenMDW-1.1 license, and the imagery is EUMETSAT
  Meteosat FCI Level 1c data. Check both before sharing anything you train.
