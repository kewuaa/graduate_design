import torch
from torch import nn
from torch.nn import functional as F

from . import lovasz_losses
from ....utils.dice import dice_coeff


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
            self._loss_func = nn.CrossEntropyLoss()
            self.forward = self._multiple
        else:
            self._loss_func = nn.BCEWithLogitsLoss()
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
            torch.sigmoid(input),
            target,
            reduce_batch_first=True
        )
        return loss
