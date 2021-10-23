'''pytorch models'''
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .PAE import PAE
from .PAE import vallina_mse_loss_function
from .PAE import PAEDataset
from numpy import ndarray
from torch import tensor
from typing import Tuple
from torch.optim import Optimizer
from tqdm import tqdm

class DimensionReducer:
    """Reduce dimsions with Positional Autoencoder"""
    def __init__(self, expressionData: ndarray, spatialMatrix: tensor, zdim: int, PEalpha: float, epochs : int =501):
        self.model, self.train_loader, self.optimizer, self.device = self.setup(expressionData, zdim, PEalpha)
        self.spatialMatrix = spatialMatrix
        self.epochs = epochs

    def setup(self, expressionData: ndarray, zdim: int, PEalpha: float, cuda : bool = False, batch_size : int = 12800, seed: int = 1)->Tuple[PAE, DataLoader, Optimizer, torch.device]: 
        """Set-up GNN"""

        torch.manual_seed(seed)
        device = torch.device("cuda" if cuda else "cpu")
        torch.set_num_threads(1)
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

        #Fix data-loader
        expressionDataset = PAEDataset(expressionData)
        train_loader = DataLoader(
            expressionDataset, batch_size=batch_size, shuffle=False, **kwargs)
        #Fix model & optimizer
        model = PAE(D=zdim, qlayers=5, qdim=512, players=5, 
                    pdim=512, in_dim=expressionDataset.features.shape[1], zdim=zdim, 
                    outdim=expressionDataset.features.shape[1],
                    pe_alpha=PEalpha).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        return model, train_loader, optimizer, device

    def reduceDims(self)->ndarray:
        """Train GNN and output embedding"""
        for epoch in tqdm(range(1, self.epochs)):
            self.model.train()
            train_loss = 0
            for batch_idx, (data, dataindex) in enumerate(self.train_loader):
                #(cells, genes)
                data = data.type(torch.FloatTensor)
                data = data.to(self.device)
                
                #Get (x,y)
                spatialMatrixBatch = self.spatialMatrix[dataindex, :]
                spatialMatrixBatch = spatialMatrixBatch.to(self.device)

                self.optimizer.zero_grad()

                B = data.size(0)
                # encode gene expressions
                z = self.model.encode(data)
                # decode
                recon_batch = self.model(spatialMatrixBatch, z).view(B, -1)

                mu_dummy, logvar_dummy = '', ''

                loss = vallina_mse_loss_function(recon_batch, 
                                           data.view(-1, recon_batch.shape[1]), 
                                           gammaPara=0.1,
                                            )
                l1 = 0.0
                for p in self.model.parameters():
                    l1 = l1 + p.abs().sum()

                loss = loss + l1
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                if batch_idx == 0:
                    z_all = z
                else:
                    z_all = torch.cat((z_all, z), 0)
        return z_all.detach().cpu().numpy()