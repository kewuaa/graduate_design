from dataclasses import dataclass
from pathlib import Path

try:
    import rtoml as toml
except ImportError:
    import tomli as toml


@dataclass(order=False, eq=False)
class Config:
    epoch_num: int = 5
    batch_size: int = 1
    learning_rate: float = 1e-5
    validation_percent: float = 0.1
    weight_decay: float = 1e-8


def load() -> Config:
    config_file = Path('./config.toml')
    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = toml.load(f)
    return Config(**config)
