from pathlib import Path

import torch

from ..logging import logger


class BaseNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._device = 'cpu'
        self._checkpoint_dir = Path('./checkpoints')
        self._checkpoint_dir.mkdir(exist_ok=True)

    def load(self, path: Path) -> None:
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def save(self, name: str) -> None:
        path = self._checkpoint_dir / f'checkpoint_{name}.pth'
        torch.save(self.state_dict(), str(path))

    def start_train(self, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda':
            if not torch.cuda.is_available():
                logger.warn('cuda is not available in your computer')
                device = 'cpu'
        self._device = torch.device(device)

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
