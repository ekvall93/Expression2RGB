import time
import random
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
from .GAE import GCNModelVAE, loss_function, GAEDataPreprocess
from tqdm import tqdm

class GAEembedder(GAEDataPreprocess):
    def __init__(self, GAEhidden1 :int = 32, GAEhidden2: int = 3, GAEdropout: float = 0, GAElr: float = 0.01, GAEepochs: int = 200):
        GAEDataPreprocess.__init__(self)
        self.GAEhidden1 = GAEhidden1
        self.GAEhidden2 = GAEhidden2
        self.GAEdropout = GAEdropout
        self.GAElr = GAElr
        self.GAEepochs = GAEepochs

    def train(self, z, adj):
        '''
        GAE embedding for clustering
        Param:
            z,adj
        Return:
            Embedding from graph
        '''   

        np.random.seed(42)
        torch.manual_seed(42)

        features, adj_norm, adj_label, n_nodes, feat_dim, norm, pos_weight = self.preprocess(z, adj)

        model = GCNModelVAE(feat_dim, self.GAEhidden1, self.GAEhidden2, self.GAEdropout)
        optimizer = optim.Adam(model.parameters(), lr=self.GAElr)

        hidden_emb = None
        for epoch in tqdm(range(self.GAEepochs)):
            t = time.time()
            model.train()
            optimizer.zero_grad()

            
            z, mu, logvar = model(features, adj_norm)
            loss = loss_function(preds=model.dc(z), labels=adj_label,
                                mu=mu, logvar=logvar, n_nodes=n_nodes,
                                norm=norm, pos_weight=pos_weight)
            loss.backward()
            cur_loss = loss.item()
            optimizer.step()

            hidden_emb = mu.data.numpy()
            ap_curr = 0
        return hidden_emb
