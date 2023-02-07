from pathlib import Path
import json
default = {
    'epoch': 5,
    'batch_size': 1,
    'learning_rate': 1e-5,
    'validation_percent': 0.1,
}


def load() -> dict:
    config_file = Path('./config.json')
    default_config = default.copy()
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        default_config.update(config)
    return default_config
