from functools import partial

import cv2
import numpy as np
import torch
from rich.progress import Progress
from torch import (
    Tensor,
    autocast,
    channels_last,
    cuda,
    inference_mode,
    nn,
    sigmoid,
    optim,
    where,
)
from torch.utils.data import DataLoader, random_split
from torchnet import meter

from ...data import Dataset
from ...utils.visdom import Visualizer
from ..base import BaseNet
from .unet_parts import DoubleConv, Down, OutConv, Up


class UNet(BaseNet):
    def __init__(self, n_classes, bilinear=False):
        super(UNet, self).__init__(name='unet')
        scale = self._config.scale
        self.n_classes = n_classes
        self._unique_values = None
        self._dataset = Dataset(
            self._config.batch_size,
            pre_process=self.pre_process
        )
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
        self.outc = (OutConv(64, n_classes))

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
        if not hasattr(self, '_unique_values'):
            self._unique_values = np.unique(label)
        for i, v in enumerate(self._unique_values):
            label[label == v] = i
        img = (img / 255).astype(np.float32)
        return np.expand_dims(img, axis=0), label.astype(np.int64)

    def __dice_coeff(
        self,
        input: Tensor,
        target: Tensor,
        reduce_batch_first: bool = False,
        epsilon: float = 1e-6
    ):
        sum_dim = (-1, -2) \
            if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()

    def __multiclass_dice_coeff(
        self,
        input: Tensor,
        target: Tensor,
        reduce_batch_first: bool = False,
        epsilon: float = 1e-6
    ):
        # Average of Dice coefficient for all classes
        return self.__dice_coeff(
            input.flatten(0, 1),
            target.flatten(0, 1),
            reduce_batch_first,
            epsilon
        )

    def _dice_loss(
        self,
        input: Tensor,
        target: Tensor,
        multiclass: bool = False
    ):
        # Dice loss (objective to minimize) between 0 and 1
        fn = self.__multiclass_dice_coeff if multiclass else self.__dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)

    def load(self, path) -> None:
        state_dict = torch.load(path)
        self._unique_values = state_dict.pop('unique_values', None)
        self._config = state_dict.pop('config', None)
        self.load_state_dict(state_dict)

    def save(self, suffix: str = ''):
        path = self._checkpoint_dir / f'checkpoint_{self._name + suffix}.pth'
        state_dict = self.state_dict()
        if self._unique_values is not None:
            state_dict['unique_values'] = self._unique_values
        if self._config is not None:
            state_dict['config'] = self._config
        torch.save(state_dict, path)

    def start_train(self, device: str = None):
        super().start_train(device)
        device = self._device
        self.to(device)

        # load config
        config = self._config
        epoch_num = config.epoch_num
        batch_size = config.batch_size
        validation_percent = config.validation_percent
        learning_rate = config.learning_rate
        weight_decay = config.weight_decay
        momentum = config.momentum
        gradient_clipping = config.gradient_clipping
        amp = config.amp and device.type == 'cuda'

        # 1. Create dataset
        dataset = self._dataset

        # 2. Split into train / validation partitions
        length = len(dataset)
        validate_set_num = int(length * validation_percent)
        train_set_num = length - validate_set_num
        train_set, validate_set = random_split(
            dataset,
            [train_set_num, validate_set_num]
        )

        # 3. Create dataloaders
        train_loader = DataLoader(train_set, batch_size)
        validate_loader = DataLoader(validate_set, batch_size)

        # 4. Set up the optimizer, the loss, the learning rate scheduler
        # and the loss scaling for AMP
        optimizer = optim.RMSprop(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            foreach=True,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'max',
            patience=5,
        )
        grad_scaler = cuda.amp.GradScaler(enabled=amp)
        loss_func = nn.CrossEntropyLoss() \
            if self.n_classes > 1 else nn.BCEWithLogitsLoss()

        # 指标
        loss_meter = meter.AverageValueMeter()

        visualizer = Visualizer()
        progress = Progress()
        load_task = progress.add_task(
            '[red]load image',
            start=False
        )
        dataset.add_refresh(partial(
            progress.update,
            load_task,
            advance=1
        ))
        with progress, dataset:
            progress.update(load_task, total=length)
            progress.start_task(load_task)
            epoch_task = progress.add_task(
                '[blue]train epoch progress',
                total=epoch_num,
            )
            train_task = progress.add_task(
                '',
                total=train_set_num
            )
            evaluate_task = progress.add_task(
                '',
                total=validate_set_num,
                visible=False
            )

            division_step = train_set_num // (5 * batch_size)
            global_step = 0
            self.train()
            for epoch in range(1, epoch_num + 1):
                progress.reset(train_task)
                progress.update(
                    train_task,
                    description=f'[yellow]train epoch {epoch}'
                )

                loss_meter.reset()
                for step, batch in enumerate(train_loader):
                    image, label = batch
                    size = image.size(0)
                    image = image.to(
                        device=device,
                        memory_format=channels_last
                    )
                    label = label.to(device=device)

                    with autocast(device.type, enabled=amp):
                        pre = self(image)
                        if self.n_classes == 1:
                            pre = pre.squeeze(1)
                            label = label.float()
                            loss = loss_func(pre, label)
                            loss += self._dice_loss(
                                sigmoid(pre),
                                label,
                                multiclass=False,
                            )
                        else:
                            label = label.long()
                            loss = loss_func(pre, label)
                            loss += self._dice_loss(
                                nn.functional.softmax(pre, dim=1).float(),
                                nn.functional.one_hot(label, self.n_classes) \
                                    .permute(0, 3, 1, 2).float(),
                                multiclass=True,
                            )
                    loss_value = loss.item()
                    loss_meter.add(loss_value)
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(
                        self.parameters(),
                        gradient_clipping
                    )
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    global_step += 1

                    progress.update(train_task, advance=size)
                    visualizer.plot(step, loss_value, f'epoch {epoch}')

                    if division_step and global_step % division_step == 0:
                        progress.update(
                            evaluate_task,
                            visible=True,
                            description=f'validation of epoch {epoch}'
                        )
                        val_score = self.evaluate(
                            validate_loader,
                            device,
                            amp,
                            partial(progress.update, evaluate_task)
                        )
                        scheduler.step(val_score)
                        progress.update(evaluate_task, visible=False)
                        visualizer.log(f'Validation Dice score: {val_score}')
                progress.update(epoch_task, advance=1)
                average_loss, std_loss = loss_meter.value()

                visualizer.log(f'''
                    epoch {epoch}:<br>
                    ----train loss    : {average_loss}
                ''')
                self.save(suffix='epoch' + epoch)

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
                    pre = (sigmoid(pre) > 0.5).float()
                    # compute the Dice score
                    dice_score += self.__dice_coeff(
                        pre,
                        label,
                        reduce_batch_first=False
                    )
                else:
                    label = label.long()
                    # convert to one-hot format
                    label = nn.functional.one_hot(
                        label,
                        self.n_classes
                    ).permute(0, 3, 1, 2).float()
                    pre = nn.functional.one_hot(
                        pre.argmax(dim=1),
                        self.n_classes
                    ).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += self.__multiclass_dice_coeff(
                        pre[:, 1:],
                        label[:, 1:],
                        reduce_batch_first=False
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
        label = cv2.resize(label, self._origin_size, None, 0., 0., cv2.INTER_NEAREST)
        return img, label, pre

    @inference_mode()
    def predict(self, img, process: bool = True):
        if process:
            if type(img) is str:
                img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            img = (cv2.resize(img, self._new_size) / 255).astype(np.float32)
            img = Tensor(np.expand_dims(img, axis=0)).contiguous()
            img = img.unsqueeze(0)
        pre = self(img)
        pre = nn.functional.interpolate(pre, self._origin_size, mode='bilinear')
        if self.n_classes > 1:
            pre = pre.argmax(dim=1)
        else:
            pre = sigmoid(pre) > 0.5
        pre = pre.squeeze().numpy()
        for i, v in enumerate(self._unique_values):
            pre[pre == i] = v
        return pre
