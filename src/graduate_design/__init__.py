import torch

from . import model
from .data import dataset


def generate_data() -> None:
    dataset.init()


def train(model_name: str, *, device: str = 'cpu'):
    net: model.base.BaseNet = None
    if model_name == 'automap':
        net = model.Automap()
    elif model_name == 'unet':
        net = model.UNet()
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
