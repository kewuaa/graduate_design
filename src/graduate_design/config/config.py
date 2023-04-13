from typing import Union
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
    def read_toml(path: str) -> dict:
        with open(path, 'rb') as f:
            content = tomllib.load(f)
        return content
except ImportError:
    import rtoml
    def read_toml(path: str) -> dict:
        with open(path) as f:
            content = rtoml.load(f)
        return content
config_file = Path('./config.toml')
__config = {}


@dataclass(order=False, eq=False)
class DataConfig:
    image_num: int = 10000
    image_size: int = 64
    pixel: Union[tuple, int] = 0
    circle_num: Union[tuple, int] = 3
    circle_size: Union[tuple, int] = 10
    theta_step: float = 0.
    angle: tuple = (0, 180)
    noise: bool = True

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
    optimizer: str = 'rms'
    momentum: float = 1.
    betas: tuple = (0.9, 0.9)
    nesterov: bool = True
    gradient_clipping: float = 1.
    amp: bool = True


@dataclass(order=False, eq=False)
class AutomapConfig(ModelConfig):
    alpha: float = 0.9


@dataclass(order=False, eq=False)
class UnetConfig(ModelConfig):
    n_classes: int = 0
    unique_values: tuple = None
    loss: str = 'normal'


def init():
    global __config
    if config_file.exists():
        config = read_toml(config_file)
    data = DataConfig(**config.get('data', {}))
    automap = AutomapConfig(**config.get('automap', {}))
    unet = UnetConfig(**config.get('unet', {}))
    if not (unet.n_classes and unet.unique_values):
        if type(data.pixel) is int:
            if unet.n_classes:
                assert unet.n_classes == 2
            else:
                unet.n_classes = 1
            unet.unique_values = [data.pixel]
        else:
            unet.unique_values = list(range(*data.pixel))
            unet.n_classes = len(unet.unique_values) + 1
            if unet.n_classes == 2:
                data.pixel = unet.unique_values[0]
    __config = {
        'data': data,
        'automap': automap,
        'unet': unet,
    }


def get(type_: str) -> dict:
    return __config.get(type_)


init()
