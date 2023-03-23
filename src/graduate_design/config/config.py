from typing import Union
from dataclasses import dataclass
from pathlib import Path

try:
    import rtoml as toml
except ImportError:
    import tomli as toml


@dataclass(order=False, eq=False)
class DataConfig:
    image_num: int = 10000
    image_size: int = 64
    pixel: Union[tuple, int] = 0
    circle_num: Union[tuple, int] = 3
    circle_size: Union[tuple, int] = 10
    theta_step: float = 0.
    angle: tuple = (0, 180)
    reinit: bool = False

    def __post_init__(self):
        if not self.theta_step:
            self.theta_step = 180 / self.image_size


@dataclass(order=False, eq=False)
class ModelConfig:
    scale: float = 1.
    epoch_num: int = 5
    batch_size: int = 1
    learning_rate: float = 1e-5
    validation_percent: float = 0.1
    weight_decay: float = 1e-8
    amp: bool = True


@dataclass(order=False, eq=False)
class AutomapConfig(ModelConfig):
    alpha: float = 0.9


@dataclass(order=False, eq=False)
class UnetConfig(ModelConfig):
    momentum: float = 1.
    gradient_clipping: float = 1.


__config_file = Path('./config.toml')
__config = {
    'data': {},
    'automap': {},
    'unet': {},
}
if __config_file.exists():
    with open(__config_file) as f:
        __config = toml.load(f)
__config = {
    'data': DataConfig(**__config['data']),
    'automap': AutomapConfig(**__config['automap']),
    'unet': UnetConfig(**__config['unet'])
}


def get(type_: str) -> dict:
    return __config.get(type_)
