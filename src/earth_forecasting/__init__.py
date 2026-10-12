"""Satellite nowcasting over Ireland with NVIDIA's StormScope, fine-tuned with LoRA.

Modules:
    stormscope  load the model, crop it to a region, change its time step
    data        download MTG-FCI frames and serve training samples
    train       LoRA fine-tuning driven by configs/train.yaml
    forecast    run a forecast from live EUMETSAT data
"""
