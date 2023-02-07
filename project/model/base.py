from pathlib import Path
import time

import torch


class BaseNet(torch.nn.Module):
    def load(self, path: Path) -> None:
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def save(self, path: Path = None) -> None:
        if path is None:
            path = Path('./checkpoints/')
            path.mkdir(exist_ok=True)
            path /= time.strftime(f'{str(type(self))}_%m%d_%H:%M:%S.pth')
        torch.save(self, path)
