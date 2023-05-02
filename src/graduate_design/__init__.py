import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import iradon
from skimage.metrics import (
    peak_signal_noise_ratio as PSNR,
    structural_similarity as SSIM
)

from . import model
from . import config
from .data import dataset
from .utils.tools import timer
iradon = timer(iradon)
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


def validate(checkpoint_index: int = None, device: str = None):
    net.auto_load(checkpoint_index, device)
    img, label, pre = net.validate()
    thetas = np.arange(0., 180., step=config.theta_step)
    iradon_img = iradon(img, theta=thetas, filter_name='ramp')
    psnr_unet = PSNR(label, pre)
    ssim_unet = SSIM(label, pre)
    psnr_fbp = PSNR(label, iradon_img)
    ssim_fbp = SSIM(label, pre)
    print(f'''
    PSNR of FBP is {psnr_fbp}
    SSIM of FBP is {ssim_fbp}
    PSNR of U-NET is {psnr_unet}
    SSIM of U-NET is {ssim_unet}
    ''')
    plt.subplot(141)
    plt.title('sinogram')
    plt.imshow(img, cmap='gray')
    plt.subplot(142)
    plt.title('real image')
    plt.imshow(label, cmap='gray')
    plt.subplot(143)
    plt.title('FBP')
    plt.imshow(iradon_img, cmap='gray')
    plt.subplot(144)
    plt.title('predicted image')
    plt.imshow(pre, cmap='gray')
    plt.show()


def predict(img_path: str, checkpoint_index: int = None, device: str = None):
    net.auto_load(checkpoint_index, device)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pre = net.predict(img)
    thetas = np.arange(0., 180., step=config.theta_step)
    iradon_img = iradon(img, theta=thetas, filter_name='ramp')
    plt.subplot(131)
    plt.title('sinogram')
    plt.imshow(img, cmap='gray')
    plt.subplot(132)
    plt.title('FBP')
    plt.imshow(iradon_img, cmap='gray')
    plt.subplot(133)
    plt.title('predicted image')
    plt.imshow(pre, cmap='gray')
    plt.show()
