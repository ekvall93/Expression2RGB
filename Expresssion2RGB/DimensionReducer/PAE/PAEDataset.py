from torch.utils.data import Dataset
import torch
import scipy.sparse as sp

class PAEDataset(Dataset):
    def __init__(self, data=None):
        """Create GNN dataset"""
        # Now lines are cells, and cols are genes
        self.features = data.transpose()
        
    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #Pick cells
        sample = self.features[idx, :]
        if type(sample) == sp.lil_matrix:
            sample = torch.from_numpy(sample.toarray())
        else:
            sample = torch.from_numpy(sample)

        return sample, idx
