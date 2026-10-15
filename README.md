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

## Running it

There is a single command, and the config decides what it does:

```bash
uv run earth configs/train.yaml       # download the data, train, evaluate
uv run earth configs/evaluate.yaml    # download the test period, evaluate an existing adapter
```

Each config has a `steps` list, run in order: `download`, `check`, `train`, `evaluate`,
`forecast` and `publish`. `train.yaml` does `[download, train, evaluate]`, and `evaluate.yaml` does
`[download, evaluate]`. You can pick other steps from the command line, and override any other
value in the same way:

```bash
uv run earth configs/evaluate.yaml steps=[forecast] forecast.start=2026-10-12T15:00
uv run earth configs/train.yaml steps=[train] train.epochs=4 lora.rank=16
```

Before running anything, the command checks that the config has every section the requested steps
need, so a typo fails straight away rather than after hours of downloading. `uv run earth --help`
lists the steps.

## Downloading data

The `download` step fetches every period in the config's `data` section (`train`, `val`, and
`test` if there is one), including the hours before and after each period that the samples need.
Frames already on disk are skipped, so you can stop it and start it again at any time.

Plan for the download time. EUMETSAT only serves full-disk products, 0.5 to 1 GB each, so every
hour of data means downloading the whole product, cropping out Ireland and throwing the rest away.
That has taken anywhere from 20 seconds to over a minute per hour of data, depending on how busy
EUMETSAT's servers are; each saved frame takes only 1.8 MB of disk. The default training period
(June to August, plus a week of validation) is about 2,400 frames, so expect a day or more.

Run one download at a time. Reading a product briefly takes about 15 GB of memory, and on a 32 GB
machine two or three downloads in parallel run it out of memory.

You can stop the download at any time. Frames that are already saved get skipped on the next run,
and an hour that EUMETSAT can't serve is just reported and skipped.

## Checking the data

Before training on anything, it's worth checking that the frames are complete and look like what
the model expects:

```bash
uv run earth configs/train.yaml steps=[check]
```

This runs every training and validation window through a DataLoader and through the same input
transform the StormScope authors use (radiances to raw counts, normalized with their per-channel
means and standard deviations). For each channel it prints the mean and spread of the result,
the share of missing pixels, the share of values outside the instrument's range, and how many
frames have a constant channel. The authors normalized over all of Europe, so for Ireland you
should see means and spreads of order one rather than exactly zero and one.

The check matters because the training code turns missing pixels into zeros, and the model would
quietly learn from those holes. If any channel misses more than 1% of its pixels the check stops
with an error, and training runs the same check before it starts. It also saves one example
window to `runs/data_check.png`, so you can see that the frames look like real cloud fields.

## Training

The `train` step fine-tunes the model on the config's training period:

```bash
uv run earth configs/train.yaml steps=[train]
```

By default only the attention layers get adapters. Setting `lora.output_layer: true` also adapts
the last layer of each expert, the projection that turns tokens back into the sixteen channels.
It adds very few parameters, and it's the natural place to fix errors that differ from channel to
channel. It still uses LoRA rather than training the layer outright, so the evaluation can switch
every adapter off and compare against the untouched base model.

Any config value can be overridden from the command line without editing the file:

```bash
uv run earth configs/train.yaml train.epochs=4 lora.rank=16
```

The batch size is 1, because that's what fits in memory, and gradients are accumulated over 8
samples before each optimizer step. Each sample trains one expert, alternating between the two, at
a noise level drawn from that expert's range. On an RTX 5060 Laptop a sample takes about half a
second and the run peaks around 4 GB of GPU memory.

The optimizer is [Muon](https://kellerjordan.github.io/posts/muon/) by default, PyTorch's built-in
`torch.optim.Muon`. It works on 2-D weight matrices, which is exactly what LoRA adapters are, and it
orthogonalizes each update so that no single direction dominates. Its updates are rescaled to the
size AdamW's would have, so the same learning rate works for both, and you can switch back with
`train.optimizer=adamw`. Muon is well tested on full transformer training and much less on LoRA
adapters, so comparing the two is worthwhile.

The learning rate warms up linearly over the first 5% of optimizer steps, then follows a cosine
curve down to 10% of its peak. Gradients are clipped to a norm of 1, and a small weight decay pulls
the adapters gently back toward the base model. All of this is in the `train` section of the
config. The length of the schedule comes from `train.epochs`, the number of passes over the
training windows; `train.max_hours` is only a safety limit.

Training is organized in epochs, one pass over every training window. After each epoch the
model is validated with fixed noise, so the numbers are comparable from one epoch to the next, and
a line like `epoch 3/8  train 0.4312  val 0.4219  12.4 min` is printed. Epoch 0 is the base model,
before any training, which gives you the baseline. `best.pt` is only written when an epoch beats
it; `last.pt` is always written at the end. Both go to `checkpoints/lora_1h/`.

To watch progress:

```bash
uv run tensorboard --logdir runs
```

Every curve has one point per epoch: the mean training loss (overall and for each expert), the
validation loss, the learning rate at the end of the epoch, the mean gradient norm, minutes per
epoch and peak GPU memory. The config the run used is saved under the Text tab.

## Evaluating forecasts

The validation loss tells you whether the denoiser is improving, but not directly whether the
forecasts are any good. So the `evaluate` step, which `train.yaml` runs after training, makes real forecasts at times
spread over the validation period, one and two hours ahead (two hours means predicting one hour
and feeding that prediction back in). Each forecast is compared with what the satellite actually
saw, and so are two references:

- **base**: the same model with the LoRA adapters switched off, which is where you started.
- **persistence**: assuming the next hour looks exactly like the last one. This sounds naive, but
  over an hour or two it's hard to beat, and a forecast that doesn't is not worth much.

The score is the RMSE of the normalized channels over the land and sea pixels, printed per
channel and lead time and logged to TensorBoard. Both models use the same diffusion noise for
each date, so the difference between them comes from the adapter alone. You can also run the
evaluation by itself, on the best adapter or any other:

```bash
uv run earth configs/train.yaml steps=[evaluate]
uv run earth configs/train.yaml steps=[evaluate] adapter=checkpoints/lora_1h/last.pt
```

Without `adapter=...` it scores `best.pt` from the config's checkpoint folder, or `last.pt` if
there is no `best.pt`.

The `eval` section of the config sets the lead times, the number of dates (20 by default) and
the diffusion steps. Every forecast step runs the full 48-step sampler, so the evaluation takes
a while.

### Evaluating without training

[`configs/evaluate.yaml`](configs/evaluate.yaml) is for everything that doesn't train: scoring an
adapter you already have on a new period, and making forecasts with it. It has a `test` period
(the first week of October by default, after every training and validation date) and, at the top,
the `adapter` to use:

```bash
uv run earth configs/evaluate.yaml
```

The download fetches only the test period, including the hours before and after it that the
forecasts need. Whenever a config has a `test` period, the evaluation uses it instead of `val`.
The region and step have to match the config the adapter was trained with.

## Publishing to Hugging Face

When you have an adapter you're happy with, set `hub.repo_id` in the config (for example
`your-name/stormscope-ireland-lora`), log in once, and publish:

```bash
uv run hf auth login
uv run earth configs/train.yaml steps=[publish]
```

This uploads the adapter (`best.pt`, or `last.pt` if validation never improved), the exact config it
was trained with, and a model card describing the setup. The repository is created as private
unless you set `hub.private: false`.

## Forecasting

```bash
uv run earth configs/evaluate.yaml steps=[forecast]
uv run earth configs/evaluate.yaml steps=[forecast] forecast.start=2026-10-12T15:00 forecast.steps=3
```

This makes one forecast from `forecast.start` (UTC). The six frames before it are downloaded the
same way as the training data, or read from disk if they're already there, and the model then
forecasts `forecast.steps` hours ahead, feeding each prediction back in. It uses the config's
`adapter`; with `adapter=null` it runs the base model instead. The predicted radiances are saved to
`runs/forecasts/` as one `.npy` array of shape (steps, channels, y, x).

## Smoke tests

Each main module has a quick self-check you can run on its own. They use synthetic data, so they
don't need EUMETSAT credentials or a network connection (except for the first download of the
model weights):

```bash
uv run python -m earth_forecasting.data        # building training windows from frames on disk
uv run python -m earth_forecasting.checks      # clean frames pass the data check, missing pixels don't
uv run python -m earth_forecasting.stormscope  # loading the model and running it on the crop
uv run python -m earth_forecasting.train       # data check, a few training steps, checkpoints, logs
uv run python -m earth_forecasting.evaluate    # +1h/+2h forecasts scored against base and persistence
uv run python -m earth_forecasting.forecast    # a two-step forecast from frames on disk, saved as .npy
uv run python -m earth_forecasting.hub         # preparing the upload folder, without uploading
```

All of them except `data` and `hub` need the GPU.

## Project layout

```
configs/train.yaml          every setting for downloading and training
configs/evaluate.yaml       everything except training: evaluation and forecasts
src/earth_forecasting/
    stormscope.py           load the model, crop it to a region, change its time step
    cli.py                  the `earth` command: runs the config's steps, in order
    config.py               load a config from configs/, with key=value overrides
    data.py                 download frames and turn them into training samples
    checks.py               check downloaded frames before training
    train.py                LoRA fine-tuning
    evaluate.py             forecast skill against the base model and persistence
    forecast.py             forecasts from live EUMETSAT data
    hub.py                  publish an adapter to Hugging Face
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
