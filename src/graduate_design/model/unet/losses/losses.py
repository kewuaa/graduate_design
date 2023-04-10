import torch
from torch import nn, Tensor
from torch.nn import functional as F

from . import lovasz_losses


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


class LovaszLoss(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.forward = self._softmax if n_classes > 1 else self._hinge

    def _softmax(self, input, target):
        return lovasz_losses.lovasz_softmax(F.softmax(input, dim=1), target)

    def _hinge(self, input, target):
        return lovasz_losses.lovasz_hinge(input, target)


class NormalLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        if n_classes > 1:
            self._loss_func = nn.CrossEntropyLoss
            self.forward = self._multiple
        else:
            self._loss_func = nn.BCEWithLogitsLoss
            self.forward = self._single

    def _multiple(self, input, target):
        loss = self._loss_func(input, target)
        loss += 1 - dice_coeff(
            F.softmax(input, dim=1).flatten(0, 1),
            target.flatten(0, 1),
            reduce_batch_first=True
        )
        return loss

    def _single(self, input, target):
        loss = self._loss_func(input, target)
        loss += 1 - dice_coeff(
            F.sigmoid(input),
            target,
            reduce_batch_first=True
        )
        return loss
