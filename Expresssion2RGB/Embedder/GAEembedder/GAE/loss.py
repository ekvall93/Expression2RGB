import torch
import torch.nn.modules.loss
import torch.nn.functional as F

def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    """Loss function"""
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=labels * pos_weight)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD
