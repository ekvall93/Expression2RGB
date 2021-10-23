import torch
import torch.nn as nn
from torch import Tensor
from .clusterMLP import DecoderMLP


class FTPositionalDecoder(nn.Module):
    def __init__(self, in_dim: int, D: int, nlayers: int, hidden_dim: int, pe_alpha: float, outdim:bool=None):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 2
        self.zdim = in_dim - 2
        self.D = D
        self.enc_dim = D // 2 #D//2
        self.in_dim = int(2 * (self.enc_dim) * 2 + self.zdim)
        self.alpha = pe_alpha
        self.decoder = DecoderMLP(self.zdim, outdim)

    def positional_encoding(self, coords: Tensor)->Tensor:
        """Positional encoding"""
        freqs = torch.arange(1, self.enc_dim+1, dtype=torch.float)
        freqs = freqs.view(*[1]*len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 2 x 1
        k = coords[..., 0:2, :] * freqs  # B x 2 x D2
        s = torch.zeros((k.shape[0],k.shape[1],k.shape[2]))  # B x 2 x D2
        c = torch.zeros((k.shape[0],k.shape[1],k.shape[2]))  # B x 2 x D2
        x = torch.cat([s, c], -1)  # B x 2 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        tmp = torch.add(x[:,:self.D], coords[..., 2:, :].squeeze(-1),alpha=self.alpha)
        x = torch.add(tmp, x[:,self.D:],alpha=self.alpha)
        return x

    def forward(self, coords: Tensor)->Tensor:
        '''Input should be coordinates from [-.5,.5]'''
        assert (coords[..., 0:2].abs() - 0.5 < 1e-4).all()
        return self.decoder(self.positional_encoding(coords))