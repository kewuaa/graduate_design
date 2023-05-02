import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import iradon, iradon_sart

from . import model
from . import config
from .data import dataset
config = config.get('data')
generate_data = dataset.init
net = None


def load_model(model_name: str):
    global net
    if model_name == 'automap':
        net = model.Automap()
    elif model_name == 'unet':
        net = model.UNet()
    else:
        raise



def train(*, device: str = None):
    try:
        net.start_train(device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        net.use_checkpoint()
        net.start_train(device)


def validate(checkpoint_index: int = None):
    net.auto_load(checkpoint_index)
    img, label, pre = net.validate()
    thetas = np.arange(0., 180., step=config.theta_step)
    iradon_img = iradon(img, theta=thetas, filter_name='ramp')
    iradon_sart_img = iradon_sart(img, theta=thetas)
    plt.subplot(131)
    plt.title('sinogram')
    plt.imshow(img, cmap='gray')
    plt.subplot(132)
    plt.title('real image')
    plt.imshow(label, cmap='gray')
    plt.subplot(133)
    plt.title('FBP')
    plt.imshow(iradon_img, cmap='gray')
    plt.subplot(134)
    plt.title('SART')
    plt.imshow(iradon_sart_img, cmap='gray')
    plt.subplot(135)
    plt.title('predicted image')
    plt.imshow(pre, cmap='gray')
    plt.show()


def predict(img_path: str, checkpoint_index: int = None):
    net.auto_load(checkpoint_index)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pre = net.predict(img)
    thetas = np.arange(0., 180., step=config.theta_step)
    iradon_img = iradon(img, theta=thetas, filter_name='ramp')
    iradon_sart_img = iradon_sart(img, theta=thetas)
    plt.subplot(121)
    plt.title('sinogram')
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.title('FBP')
    plt.imshow(iradon_img, cmap='gray')
    plt.subplot(123)
    plt.title('SART')
    plt.imshow(iradon_sart_img, cmap='gray')
    plt.subplot(134)
    plt.title('predicted image')
    plt.imshow(pre, cmap='gray')
    plt.show()
