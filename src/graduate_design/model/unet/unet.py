from functools import partial

import cv2
import numpy as np
import torch
from torch import (
    Tensor,
    autocast,
    channels_last,
    inference_mode,
    nn,
    optim,
)
from ..base import BaseNet
from .unet_parts import DoubleConv, Down, OutConv, Up
from . import losses


class UNet(BaseNet):
    def __init__(self, bilinear=False):
        super(UNet, self).__init__(name='unet')
        scale = self._config.scale
        self.n_classes = self._config.n_classes
        self._unique_values = self._config.unique_values
        self._origin_size = (self._dataset.img_size,) * 2
        self._new_size = (int(self._dataset.img_size * scale),) * 2

        self.inc = (DoubleConv(1, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, self.n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    def pre_process(self, data: tuple):
        img, label = data
        img = cv2.resize(img, self._new_size, None, 0., 0., cv2.INTER_CUBIC)
        label = cv2.resize(label, self._new_size, None, 0., 0., cv2.INTER_NEAREST)
        for i, v in enumerate(self._unique_values):
            label[label == v] = i + 1
        if self.n_classes > 1:
            label = np.transpose(
                np.eye(self.n_classes)[label],
                [2, 0, 1]
            )
        img = (img / 255).astype(np.float32)
        return np.expand_dims(img, axis=0), label.astype(np.int64)

    def load(self, path) -> None:
        state_dict = torch.load(path)
        self._unique_values = state_dict.pop('unique_values', None)
        self._config = state_dict.pop('config', None)
        self.load_state_dict(state_dict)

    def save(self, suffix: str = ''):
        path = self._checkpoint_dir / f'checkpoint_{suffix}.pth'
        state_dict = self.state_dict()
        if self._unique_values is not None:
            state_dict['unique_values'] = self._unique_values
        if self._config is not None:
            state_dict['config'] = self._config
        torch.save(state_dict, path)

    def start_train(self, device: str = None):
        loss_type = self._config.loss
        if loss_type == 'normal':
            loss_func = losses.NormalLoss(self.n_classes)
        elif loss_type == 'lovasz':
            loss_func = losses.LovaszLoss(self.n_classes)
        else:
            raise RuntimeError(f'bad config: invalid loss type: {loss_type}')
        super().start_train(
            loss_func,
            partial(
                optim.lr_scheduler.ReduceLROnPlateau,
                mode='min',
                patience=5
            )
        )

    @inference_mode()
    def evaluate(self, dataloader, device, amp, refresh):
        self.eval()
        num_val_batches = len(dataloader)
        dice_score = 0

        # iterate over the validation set
        for batch in dataloader:
            image, label = batch

            # move images and labels to correct device and type
            image = image.to(device=device, memory_format=channels_last)
            label = label.to(device=device)

            with autocast(device.type, enabled=amp):
                # predict the mask
                pre = self(image)

                if self.n_classes == 1:
                    pre = (torch.sigmoid(pre.squeeze(1)) > 0.5).float()
                    # compute the Dice score
                    dice_score += losses.dice_coeff(
                        pre,
                        label,
                    )
                else:
                    pre = nn.functional.one_hot(
                        pre.argmax(dim=1),
                        self.n_classes
                    ).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += losses.dice_coeff(
                        pre[:, 1:].flatten(0, 1),
                        label[:, 1:].flatten(0, 1),
                    )
            refresh(advance=image.size(0))
        self.train()
        return dice_score / max(num_val_batches, 1)

    def validate(self, index: int = None):
        img, label = self._dataset.load_one(index)
        pre = self.predict(img, process=False)
        img = img.squeeze().numpy()
        label = label.squeeze().numpy()
        img = cv2.resize(img, self._origin_size, None, 0., 0., cv2.INTER_CUBIC)
        label = cv2.resize(
            label.argmax(axis=0) if self.n_classes > 1 else label,
            self._origin_size,
            None,
            0., 0.,
            cv2.INTER_NEAREST
        )
        for i, v in enumerate(self._unique_values):
            label[label == i + 1] = v
        return img, label, pre

    @inference_mode()
    def predict(self, img, process: bool = True):
        if process:
            if type(img) is str:
                img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            img = (cv2.resize(img, self._new_size) / 255).astype(np.float32)
            img = Tensor(np.expand_dims(img, axis=0)).contiguous()
            img = img.unsqueeze(0)
        img.to(self._device)
        pre = self(img)
        pre = nn.functional.interpolate(pre, self._origin_size, mode='bilinear')
        if self.n_classes > 1:
            pre = pre.argmax(dim=1)
        else:
            pre = torch.sigmoid(pre) > 0.5
        pre = pre.cpu().squeeze().numpy()
        for i, v in enumerate(self._unique_values):
            pre[pre == i + 1] = v
        return pre
