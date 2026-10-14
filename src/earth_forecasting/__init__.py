"""Satellite nowcasting over Ireland with NVIDIA's StormScope, fine-tuned with LoRA.

Modules:
    cli         the `earth` command: one entry point that runs the config's steps
    config      load a YAML config from configs/ plus key=value overrides
    stormscope  load the model, crop it to a region, change its time step
    data        download MTG-FCI frames and serve training samples
    checks      check downloaded frames before training
    train       LoRA fine-tuning driven by configs/train.yaml
    evaluate    forecast skill against the base model and persistence
    forecast    one forecast from real satellite frames, saved to disk
    hub         publish an adapter to Hugging Face
"""
