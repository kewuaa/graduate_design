from torch import nn
import torch


class Automap(nn.Module):
    def __init__(self, projection_num: int, img_size: int) -> None:
        super(Automap, self).__init__()
        self._img_size = img_size
        self.layer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(img_size * img_size, img_size * projection_num),
            nn.Tanh(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(img_size * projection_num, img_size * img_size),
            nn.Tanh(),
            nn.Dropout(0.5),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 1, 3, padding=1),
            nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view((x.size(0), 1, self._img_size, self._img_size))
        x = self.layer3(x)
        return x
