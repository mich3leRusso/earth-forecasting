import torch
from physicsnemo.experimental.peft import (
    LoRAConfig, apply_lora, split_params_for_optimizer,
    print_trainable_parameters, save_adapter,
)

from earth_forecasting.model.StormScope import IRELAND, StormScopeMeteosatEU, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model = load_model(device, region=IRELAND)                 # any torch.nn.Module

# 1) Inject adapters into the attention projections; freeze everything else.
config=LoRAConfig(target_pattern=r"blocks\.\d+\.Attn\.(qkv_project|out_linear|cross_[qkv])")


apply_lora(model, config)
print_trainable_parameters(model)     # "trainable params: N (X% of M total)"

# 2) Train only the adapter parameters with AdamW.
groups = split_params_for_optimizer(model)
optimizer = torch.optim.AdamW(groups["lora"] + groups["extras"], lr=5e-4)
# ... standard training loop ...
# Regex (note the anchor on the block path, so only attention layers match):




# 3) Save the (small) adapter.
#save_adapter(model, "adapter.lora")