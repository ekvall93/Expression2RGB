'''pytorch models'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import EncoderMLP, FTPositionalDecoder
from torch import Tensor

class PAE(nn.Module):
    def __init__(self, D : int,
                 qlayers: int, qdim: int,
                 players: int, pdim: int,
                 in_dim: int, zdim: int = 1,
                 outdim: int = 10000,
                 pe_alpha: float = 1.0):
        super(PAE, self).__init__()
        self.D = D
        self.zdim = zdim
        self.in_dim = in_dim
        self.encoder = EncoderMLP(in_dim, zdim)
        self.pe_alpha = pe_alpha
        self.decoder = FTPositionalDecoder(
            2+zdim, D, players, pdim, pe_alpha, outdim=outdim)

    def encode(self, x: Tensor)->Tensor:
        return self.encoder(x)

    def cat_z(self, coords: Tensor, z: Tensor)->Tensor:
        '''
        coords: Bx...x3
        z: Bxzdim
        '''
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1]*(coords.ndimension()-2)), self.zdim)
        z = torch.cat(
            (coords, z.expand(*coords.shape[:-1], self.zdim)), dim=-1)
        return z

    def decode(self, coords: Tensor, z: Tensor, mask: bool = None)->Tensor:
        '''
        coords: BxNx2 image coordinates
        z: Bxzdim latent coordinate
        '''
        return self.decoder(self.cat_z(coords, z))

    def forward(self, coords: Tensor, z: Tensor)->Tensor:
        return self.decode(coords,z)