from functools import partial

from torch import (
    nn,
    cuda,
    utils,
    optim,
    Tensor,
    autocast,
    inference_mode,
    device as torch_device,
)
from torch.utils.data import DataLoader, random_split
from torchnet import meter
from rich.progress import Progress

from .base import BaseNet, RegularizeLoss
from ..data import Dataset
from ..logging import logger
from ..config import config_for_train, config_for_data
from ..utils.visdom import Visualizer


class Automap(BaseNet):
    def __init__(self) -> None:
        super(Automap, self).__init__()
        self._img_size = img_size = config_for_data.image_size
        projection_num = \
            (config_for_data.end_angle - config_for_data.start_angle) \
            / config_for_data.theta_step
        projection_num = int(projection_num)
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
        self.layer1 = utils.checkpoint(self.layer1)
        self.layer2 = utils.checkpoint(self.layer2)
        self.layer3 = utils.checkpoint(self.layer3)

    def regularize_loss(self) -> Tensor:
        return self._l1_regularizer(self._special_conv2d)

    def start_train(self, device: str = None) -> None:
        if device is None:
            device = 'cuda' if cuda.is_available() else 'cpu'
        elif device == 'cuda':
            if not cuda.is_available():
                logger.warn('cuda is not available in your computer')
                device = 'cpu'
        device = torch_device(device)
        self.to(device)

        # load config
        epoch_num = config_for_train.epoch_num
        batch_size = config_for_train.batch_size
        validation_percent = config_for_train.validation_percent
        learning_rate = config_for_train.learning_rate
        weight_decay = config_for_train.weight_decay
        amp = config_for_train.amp and device.type == 'cuda'

        # 1. Create dataset
        dataset = Dataset()

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
            alpha=0.9,
            weight_decay=weight_decay
        )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        grad_scaler = cuda.amp.GradScaler(enabled=amp)
        loss_func = nn.MSELoss()

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
                    image.to(device=device)
                    label.to(device=device)

                    with autocast(device.type, enabled=amp):
                        pre = self(image)
                        loss = loss_func(pre, label) + self.regularize_loss()
                    loss_value = loss.item()
                    loss_meter.add(loss_value)
                    optimizer.zero_grad()
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    progress.update(train_task, advance=size)
                    visualizer.plot(step, loss_value, f'epoch {epoch}')
                progress.update(epoch_task, advance=1)
                average_loss, std_loss = loss_meter.value()

                evaluate_loss = self.evaluate(validate_loader, device, amp)
                visualizer.log(f'''
                    epoch {epoch}:<br>
                    ----train loss    : {average_loss}<br>
                    ----evaluate loss : {evaluate_loss}
                ''')
                # scheduler.step(metrics)
                self.save(f'epoch_{epoch}')

    @inference_mode()
    def evaluate(self, dataloader, device, amp):
        self.eval()
        loss_meter = meter.AverageValueMeter()
        for step, batch in enumerate(dataloader):
            image, label = batch
            image.to(device=device)
            label.to(device=device)

            with autocast(device.type, enabled=amp):
                pre = self(image)
                loss = nn.functional.mse_loss(pre, label)
            loss_meter.add(loss.item())
        average_loss, std_loss = loss_meter.value()
        self.train()
        return average_loss
