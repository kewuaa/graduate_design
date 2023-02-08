from pathlib import Path
import time

import torch


class BaseNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._checkpoint_dir = Path('./checkpoints')
        self._checkpoint_dir.mkdir(exist_ok=True)

    def load(self, path: Path) -> None:
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def save(self, name: str, path: Path = None) -> None:
        if path is None:
            path = self._checkpoint_dir / time.strftime(
                f'checkpoint_{name}.pth'
            )
        torch.save(self, str(path))
