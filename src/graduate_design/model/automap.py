from functools import partial

import cv2
import torch
import numpy as np
from torch import (
    Tensor,
    autocast,
    channels_last,
    inference_mode,
    nn,
    optim,
)

from .base import BaseNet, RegularizeLoss
from ..utils.dice import dice_coeff


class Automap(BaseNet):
    def __init__(self) -> None:
        super().__init__(name='automap')
        scale = self._config.scale
        img_size = self._dataset.img_size
        projection_num = int(
            (self._dataset.angle[1] - self._dataset.angle[0]) /
            self._dataset.theta_step
        )
        self._origin_img_size = (projection_num, img_size)
        self._origin_label_size = (img_size,) * 2
        self._img_size = img_size = int(img_size * scale)
        projection_num = int(projection_num * scale)
        self._new_label_size = (img_size,) * 2
        self._new_img_size = (projection_num, img_size)
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * projection_num, img_size * projection_num),
            nn.Tanh(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(img_size * projection_num, img_size * img_size),
            nn.Tanh(),
            nn.Dropout(0.5),
        )
        self._special_conv2d = nn.Conv2d(1, 32, 3, padding=1)
        self._l1_regularizer = RegularizeLoss(1e-4, p=1)
        self.layer3 = nn.Sequential(
            self._special_conv2d,
            nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1),
            nn.ELU(),
        )

    def to(self, device):
        self._l1_regularizer.to(device)
        return super().to(device)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view((x.size(0), 1, self._img_size, self._img_size))
        x = self.layer3(x)
        return x

    def use_checkpoint(self):
        self.layer1 = torch.utils.checkpoint(self.layer1)
        self.layer2 = torch.utils.checkpoint(self.layer2)
        self.layer3 = torch.utils.checkpoint(self.layer3)

    def regularize_loss(self) -> Tensor:
        return self._l1_regularizer(self._special_conv2d)

    def pre_process(self, data: tuple):
        img, label = data
        img = cv2.resize(img, self._new_img_size, None, 0, 0, cv2.INTER_CUBIC)
        label = cv2.resize(label, self._new_label_size, None, 0, 0, cv2.INTER_NEAREST)
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        label = cv2.normalize(label, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return np.expand_dims(img, axis=0), label

    def start_train(self, device: str = None) -> None:
        loss_type = self._config.loss
        if loss_type == 'mse':
            mse_loss = nn.MSELoss()
            def loss_func(input, target):
                return mse_loss(input, target) + self.regularize_loss()
        elif loss_type == 'dice':
            def loss_func(input, target):
                input = torch.where(input > 0.5, 1, 0)
                loss = 1 - dice_coeff(input, target)
                return loss
        else:
            raise RuntimeError(f'bad config: invalid loss type: {loss_type}')
        super().start_train(
            loss_func,
            partial(
                optim.lr_scheduler.ReduceLROnPlateau,
                mode='max',
                patience=5
            )
        )

    @inference_mode()
    def evaluate(self, dataloader, device, amp, refresh):
        self.eval()
        # loss_meter = meter.AverageValueMeter()
        # for step, batch in enumerate(dataloader):
        #     image, label = batch
        #     image = image.to(device=device, memory_format=channels_last)
        #     label = label.to(device=device)

        #     with autocast(device.type, enabled=amp):
        #         pre = self(image).squeeze(1)
        #         loss = nn.functional.mse_loss(pre, label.squeeze(1))
        #     loss_meter.add(loss.item())
        #     refresh(advance=image.size(0))
        # average_loss, std_loss = loss_meter.value()
        dice_score = 0
        num_val_batches = len(dataloader)
        for batch in dataloader:
            image, label = batch
            image = image.to(device=device, memory_format=channels_last)
            label = label.to(device=device)
            with autocast(device.type, enabled=amp):
                pre = self(image).squeeze(1)
                pre = torch.where(pre > 0.5, 1, 0)
                dice_score += dice_coeff(pre, label)
            refresh(advance=image.size(0))
        self.train()
        return dice_score / max(num_val_batches, 1)

    def validate(self, index: int = None):
        img, label = self._dataset.load_one(index)
        pre = self.predict(img, process=False)
        img = img.squeeze().numpy()
        label = label.squeeze().numpy()
        img = cv2.resize(img, self._origin_img_size, None, 0, 0, cv2.INTER_CUBIC)
        label = cv2.resize(label, self._origin_label_size, None, 0, 0, cv2.INTER_NEAREST)
        return img, label, pre

    @inference_mode()
    def predict(self, img, process: bool = True):
        if process:
            if type(img) is str:
                img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self._new_img_size, None, 0, 0, cv2.INTER_CUBIC)
            img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX, cv2.CV_32F)
            img = Tensor(np.expand_dims(img, axis=0)).contiguous().unsqueeze(0)
        img.to(self._device)
        pre = self(img)
        pre = pre.cpu().squeeze().numpy()
        pre = cv2.resize(pre, self._origin_label_size, None, 0, 0, cv2.INTER_NEAREST)
        return pre > 0.5
