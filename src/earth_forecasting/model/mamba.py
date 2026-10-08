from mamba_ssm import Mamba2
from torch import nn

class VimBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fwd = Mamba2(dim, headdim=dim // 4, use_mem_eff_path=False)
        self.bwd = Mamba2(dim, headdim=dim // 4, use_mem_eff_path=False)

    def forward(self, x):  # (B, L, dim)
        n = self.norm(x)
        return x + self.fwd(n) + self.bwd(n.flip(1)).flip(1)