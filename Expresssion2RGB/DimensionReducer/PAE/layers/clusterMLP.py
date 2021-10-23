import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MLP(nn.Module):
    def __init__(self, dim: int, zdim : int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, zdim)

    def encode(self, x: Tensor)->Tensor:
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

class EncoderMLP(MLP):
    def __init__(self, dim: int, zdim : int):
        MLP.__init__(self, dim, zdim)
        self.dim = dim

    def forward(self, x: Tensor)->Tensor:
        return self.encode(x.view(-1, self.dim))

class DecoderMLP(MLP):
    def __init__(self, zdim: int, dim: int):
        MLP.__init__(self, zdim, dim)
        
    def decode(self, z: Tensor)->Tensor:
        return self.encode(z)

    def forward(self, z: Tensor)->Tensor:
        return self.decode(z)
