import torch
from torch import Tensor


def vallina_mse_loss_function(input: Tensor, target: Tensor, gammaPara : float = 0.1)->Tensor:
    r"""mse loss"""
    expanded_input, expanded_target = torch.broadcast_tensors(
        input, target)
    ret = torch._C._nn.mse_loss(
        expanded_input, expanded_target, 2)

    return gammaPara * ret
