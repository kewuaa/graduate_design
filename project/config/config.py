from dataclasses import dataclass
from pathlib import Path

try:
    import rtoml as toml
except ImportError:
    import tomli as toml


@dataclass(order=False, eq=False)
class TrainConfig:
    epoch_num: int = 5
    batch_size: int = 1
    learning_rate: float = 1e-5
    validation_percent: float = 0.1
    weight_decay: float = 1e-8
    amp: bool = False


@dataclass(order=False, eq=False)
class DataConfig:
    image_num: int
    image_size: int
    max_circle_num: int
    min_circle_size: int
    max_circle_size: int
    theta_step: float = 0.
    start_angle: int = 0
    end_angle: int = 180

    def __post_init__(self):
        if not self.theta_step:
            self.theta_step = 180 / self.image_size


config_file = Path('./config.toml')
config = {
    'train': {},
    'data': {},
}
if config_file.exists():
    with open(config_file) as f:
        config = toml.load(f)
config_for_train = TrainConfig(**config['train'])
config_for_data = DataConfig(**config['data'])
