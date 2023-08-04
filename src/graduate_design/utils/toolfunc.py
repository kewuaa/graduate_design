import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as distance
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


def onehot2dist(onehot: Tensor) -> Tensor:
    def _onehot2dist(array):
        inner_dist = distance(array)
        inner_dist[inner_dist > 0] -= 1.
        array[:] = distance(1 - array) - inner_dist
        return array
    _onehot = np.asarray(onehot)
    if _onehot.ndim > 3:
        for i in range(_onehot.shape[0]):
            _onehot2dist(_onehot[i])
    else:
        _onehot2dist(_onehot)
    return onehot
