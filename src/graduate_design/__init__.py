import cv2
import torch
from matplotlib import pyplot as plt

from . import model
from .data import dataset
generate_data = dataset.init


def train(model_name: str, *, device: str = 'cpu'):
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


def validate(model_name: str):
    if model_name == 'automap':
        net = model.Automap()
    elif model_name == 'unet':
        net = model.UNet()
    else:
        raise
    net.auto_load()
    img, label, pre = net.validate()
    plt.subplot(131)
    plt.title('sinogram')
    plt.imshow(img, cmap='gray')
    plt.subplot(132)
    plt.title('real image')
    plt.imshow(label, cmap='gray')
    plt.subplot(133)
    plt.title('predicted image')
    plt.imshow(pre, cmap='gray')
    plt.show()


def predict(model_name: str, img_path: str):
    if model_name == 'automap':
        net = model.Automap()
    elif model_name == 'unet':
        net = model.UNet()
    else:
        raise
    net.auto_load()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pre = net.predict(img)
    plt.subplot(121)
    plt.title('sinogram')
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.title('predicted image')
    plt.imshow(pre, cmap='gray')
    plt.show()
