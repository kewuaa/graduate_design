from functools import partial
from pathlib import Path

import torch
from torch import optim, cuda, channels_last, nn, autocast
from torch.utils.data import random_split, DataLoader
from torchnet import meter
from rich.progress import Progress

from ..data import Dataset
from ..logging import logger
from ..utils.visdom import Visualizer
from .. import config


class BaseNet(torch.nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._device = 'cpu'
        self._name = name
        self._config = config.get(name)
        self._dataset = Dataset(
            self._config.batch_size,
            pre_process=self.pre_process
        )
        self._checkpoint_dir = self._dataset._root_dir / 'checkpoints' / name
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def print_config(self) -> None:
        info = '\n\t\t'.join(
            f"{field}: {getattr(self._config, field)}"
            for field in self._config.__dataclass_fields__
        )
        logger.info(f'''
        train config:
                {info}
        ''')

    def auto_load(self, index: int = None):
        if index is None:
            index = self._config.epoch_num
        pth_file_path = self._checkpoint_dir / f'checkpoint_epoch{index}.pth'
        self.load(str(pth_file_path))
        self.print_config()

    def load(self, path: Path) -> None:
        state_dict: dict = torch.load(path)
        self._config = state_dict.pop('config', None)
        self.load_state_dict(state_dict)

    def save(self, suffix: str = '') -> None:
        path = self._checkpoint_dir / f'checkpoint_{suffix}.pth'
        state_dict = self.state_dict()
        if self._config is not None:
            state_dict['config'] = self._config
        torch.save(state_dict, str(path))

    def set_device(self, device: str):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda':
            if not torch.cuda.is_available():
                logger.warn('cuda is not available in your computer')
                device = 'cpu'
        logger.info(f'device set: {device}')
        self._device = torch.device(device)
        self.to(device)

    def start_train(self, loss_func, scheduler=None, device: str = None):
        self.set_device(device)
        device = self._device

        # load config
        config = self._config
        epoch_num = config.epoch_num
        batch_size = config.batch_size
        validation_percent = config.validation_percent
        learning_rate = config.learning_rate
        alpha = config.alpha
        betas = config.betas
        weight_decay = config.weight_decay
        nesterov = config.nesterov
        momentum = config.momentum
        gradient_clipping = config.gradient_clipping
        amp = config.amp and device.type == 'cuda'
        self.print_config()

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
        if config.optimizer == 'adam':
            optimizer = optim.Adam(
                self.parameters(),
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay
            )
        elif config.optimizer == 'sgd':
            optimizer = optim.SGD(
                self.parameters(),
                lr=learning_rate,
                momentum=momentum,
                nesterov=nesterov,
            )
        elif config.optimizer == 'rms':
            optimizer = optim.RMSprop(
                self.parameters(),
                lr=learning_rate,
                alpha=alpha,
                weight_decay=weight_decay,
                momentum=momentum,
                foreach=True,
            )
        else:
            raise RuntimeWarning(
                f'bad config: invalid optimizer type: {config.optimizer}'
            )
        if scheduler is not None:
            scheduler = scheduler(optimizer)
        grad_scaler = cuda.amp.GradScaler(enabled=amp)
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
                        pre = self(image).squeeze(1)
                        loss = loss_func(pre, label)
                    loss_value = loss.item()
                    loss_meter.add(loss_value)
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    if gradient_clipping > 0:
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
                        if scheduler:
                            scheduler.step(val_score)
                        progress.update(evaluate_task, visible=False)
                        visualizer.log(f'Validation loss: {val_score}')
                progress.update(epoch_task, advance=1)
                average_loss, std_loss = loss_meter.value()

                visualizer.log(f'''
                    epoch {epoch}:<br>
                    ----train loss    : {average_loss}
                ''')
                self.save(suffix=f'epoch{epoch}')

    def pre_process(self, data: tuple):
        return data

    def validate(self, index: int = None):
        pass

    def evaluate(self, dataloader, device, amp, refresh):
        pass

    def predict(self, index: int = None):
        pass


class RegularizeLoss(torch.nn.Module):
    def __init__(
        self,
        weight_decay: float,
        p: int = 2
    ) -> None:
        super().__init__()
        if weight_decay <= 0:
            raise ValueError('weight_decay is needed to be bigger then zero')
        self._weight_decay = weight_decay
        self._p = p

    def to(self, device):
        self._device = device
        return super().to(device)

    def _get_weight(self, model: torch.nn.Module) -> list:
        weight_list = [
            (name, param) for name, param in model.named_parameters()
            if 'weight' in name
        ]
        return weight_list

    def _calc_loss(self, weight_list: list) -> torch.Tensor:
        loss = sum(
            torch.norm(weight, p=self._p) for _, weight in weight_list
        )
        loss *= self._weight_decay
        return loss

    def forward(self, model: torch.nn.Module) -> torch.Tensor:
        weight_list = self._get_weight(model)
        loss = self._calc_loss(weight_list)
        return loss
