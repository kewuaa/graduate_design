import torch
from torch import nn
from torch.nn import functional as F

from . import lovasz_losses
from ...utils.toolfunc import dice_coeff, onehot2dist


class LovaszLoss(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.forward = self._softmax if n_classes > 1 else self._hinge

    def _softmax(self, input, target):
        return lovasz_losses.lovasz_softmax(F.softmax(input, dim=1), target)

    def _hinge(self, input, target):
        return lovasz_losses.lovasz_hinge(input, target)


class DiceLoss(nn.Module):
    def __init__(self, multiple: bool = 0):
        super().__init__()
        if multiple:
            self.forward = self._multiple
        else:
            self.forward = self._single

    def _multiple(self, input, target):
        loss = 1 - dice_coeff(
            F.softmax(input, dim=1).flatten(0, 1),
            target.flatten(0, 1),
            reduce_batch_first=True
        )
        return loss

    def _single(self, input, target):
        loss = 1 - dice_coeff(
            torch.sigmoid(input),
            target,
            reduce_batch_first=True
        )
        return loss


class BoundaryLoss(nn.Module):
    def forward(self, input, target):
        device = target.device
        target = onehot2dist(target.cpu().float())
        target = target.to(device)
        pc = input[:, 1:, ...].float()
        dc = target[:, 1:, ...].float()
        multiplied = pc * dc
        loss = multiplied.mean()
        return loss
