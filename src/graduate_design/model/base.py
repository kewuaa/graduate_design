from pathlib import Path

import torch

from ..data import Dataset
from ..logging import logger
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
        self._device = torch.device(device)
        self.to(device)

    def start_train(self, device: str = None):
        pass

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
