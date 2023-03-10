from functools import partial

import cv2
import torch
import numpy as np
from rich.progress import Progress
from torch import (
    Tensor,
    autocast,
    cuda,
    inference_mode,
    nn,
    optim,
)
from torch.utils.data import DataLoader, random_split
from torchnet import meter

from ..data import Dataset
from ..utils.visdom import Visualizer
from .base import BaseNet, RegularizeLoss


class Automap(BaseNet):
    def __init__(self, scale: float = 1.) -> None:
        super().__init__(name='automap')
        self._dataset = Dataset(
            self._config.batch_size,
            pre_process=self.pre_process
        )
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
        img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        label = cv2.normalize(label, None, -0.5, 0.5, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return np.expand_dims(img, axis=0), np.expand_dims(label, axis=0)

    def start_train(self, device: str = None) -> None:
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
        alpha = config.alpha
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
            alpha=alpha,
            weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            patience=5
        )
        grad_scaler = cuda.amp.GradScaler(enabled=amp)
        loss_func = nn.MSELoss()

        # ??????
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
                    image = image.to(device=device)
                    label = label.to(device=device)

                    with autocast(device.type, enabled=amp):
                        pre = self(image)
                        loss = loss_func(pre, label) + self.regularize_loss()
                    loss_value = loss.item()
                    loss_meter.add(loss_value)
                    optimizer.zero_grad()
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    global_step += 1

                    progress.update(train_task, advance=size)
                    visualizer.plot(step, loss_value, f'epoch {epoch}')

                    if division_step and global_step % division_step == 0:
                        progress.update(
                            evaluate_task,
                            description=f'validation of epoch {epoch}',
                            visible=True
                        )
                        evaluate_loss = self.evaluate(
                            validate_loader,
                            device,
                            amp,
                            refresh=partial(progress.update, evaluate_task)
                        )
                        visualizer.log(f'evaluate loss: {evaluate_loss}')
                        scheduler.step(evaluate_loss)
                        progress.update(evaluate_task, visible=False)
                progress.update(epoch_task, advance=1)
                average_loss, std_loss = loss_meter.value()

                visualizer.log(f'''
                    epoch {epoch}:<br>
                    ----train loss    : {average_loss}
                ''')
                self.save(f'automap_epoch_{epoch}')

    @inference_mode()
    def evaluate(self, dataloader, device, amp, refresh):
        self.eval()
        loss_meter = meter.AverageValueMeter()
        for step, batch in enumerate(dataloader):
            image, label = batch
            image = image.to(device=device)
            label = label.to(device=device)

            with autocast(device.type, enabled=amp):
                pre = self(image)
                loss = nn.functional.mse_loss(pre, label)
            loss_meter.add(loss.item())
            refresh(advance=image.size(0))
        average_loss, std_loss = loss_meter.value()
        self.train()
        return average_loss

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
        pre = self(img)
        pre = pre.squeeze().numpy()
        return cv2.resize(pre, self._origin_label_size, None, 0, 0, cv2.INTER_NEAREST)
