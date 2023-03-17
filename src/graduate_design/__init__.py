import torch

from . import model
from .data import dataset


def generate_data() -> None:
    dataset.init(force=True)


def train(automap: dict = None, unet: dict = None, *, device: str = 'cpu'):
    net: model.base.BaseNet = None
    if automap is not None:
        net = model.Automap(**automap)
    elif unet is not None:
        net = model.UNet(**unet)
    else:
        raise
    try:
        net.start_train(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        net.use_checkpoint()
        net.start_train(device)


def predict():
    pass
