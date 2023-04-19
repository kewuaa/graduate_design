import torch
from torch import Tensor

def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6
):
    assert input.shape == target.shape
    sum_dim = (-1, -2, -3) if reduce_batch_first else (-1, -2)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()
