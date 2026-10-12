from torch import nn
from .mamba import VimBlock


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        n = self.norm(x)
        return x + self.attn(n, n, n, need_weights=False)[0]


class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mult * dim), nn.GELU(), nn.Linear(mult * dim, dim))

    def forward(self, x):
        return x + self.net(x)


class HybridModel(nn.Module):
    # M = Vision Mamba block, * = attention, - = MLP
    def __init__(self, input_size, hidden_size, output_size, pattern="M-M-M-*-"):
        super().__init__()
        make = {"M": VimBlock, "*": Attention, "-": MLP}
        self.inp = nn.Linear(input_size, hidden_size)
        self.layers = nn.Sequential(*(make[c](hidden_size) for c in pattern))
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):  # (batch, seq_len, input_size)
        return self.fc(self.norm(self.layers(self.inp(x))))
