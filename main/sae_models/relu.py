import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# SAE model (match your training code)
# ----------------------------
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, use_relu: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.use_relu = use_relu

        self.W_e = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_e = nn.Parameter(torch.zeros(d_hidden))
        self.W_d = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_d = nn.Parameter(torch.zeros(d_in))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x @ self.W_e + self.b_e
        return F.relu(pre) if self.use_relu else pre

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        return a @ self.W_d.T + self.b_d