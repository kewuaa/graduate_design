import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import iradon
from skimage.metrics import (
    mean_squared_error as MSE,
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
    iradon_img_ = iradon(img, theta=thetas, filter_name='shepp-logan')
    iradon_img = np.uint8(
        (iradon_img - iradon_img.min()) / iradon_img.ptp() * 255
    )
    iradon_img_ = np.uint8(
        (iradon_img_ - iradon_img_.min()) / iradon_img_.ptp() * 255
    )
    mse_net = MSE(label, pre)
    psnr_net = PSNR(label, pre)
    ssim_net = SSIM(label, pre)
    mse_fbp = MSE(label, iradon_img)
    psnr_fbp = PSNR(label, iradon_img)
    ssim_fbp = SSIM(label, iradon_img)
    mse_fbp_ = MSE(label, iradon_img_)
    psnr_fbp_ = PSNR(label, iradon_img_)
    ssim_fbp_ = SSIM(label, iradon_img_)
    print(f'''
    MSE of FBP with ramp filter is {mse_fbp}
    PSNR of FBP with ramp filter is {psnr_fbp}
    SSIM of FBP with ramp filter is {ssim_fbp}
    MSE of FBP with shepp-logan filter is {mse_fbp_}
    PSNR of FBP with shepp-logan filter is {psnr_fbp_}
    SSIM of FBP with shepp-logan filter is {ssim_fbp_}
    MSE of net is {mse_net}
    PSNR of net is {psnr_net}
    SSIM of net is {ssim_net}
    ''')
    plt.subplot(151)
    plt.title('sinogram')
    plt.imshow(img, cmap='gray')
    plt.subplot(152)
    plt.title('real image')
    plt.imshow(label, cmap='gray')
    plt.subplot(153)
    plt.title('FBP with ramp filter')
    plt.imshow(iradon_img, cmap='gray')
    plt.subplot(154)
    plt.title('FBP with shepp-logan filter')
    plt.imshow(iradon_img_, cmap='gray')
    plt.subplot(155)
    plt.title('predicted image')
    plt.imshow(pre, cmap='gray')
    plt.show()


def predict(img_path: str, checkpoint_index: int = None, device: str = None):
    net.auto_load(checkpoint_index, device)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    pre = net.predict(img)
    thetas = np.arange(0., 180., step=config.theta_step)
    iradon_img = iradon(img, theta=thetas, filter_name='ramp')
    iradon_img_ = iradon(img, theta=thetas, filter_name='shepp-logan')
    plt.subplot(141)
    plt.title('sinogram')
    plt.imshow(img, cmap='gray')
    plt.subplot(142)
    plt.title('FBP with ramp filter')
    plt.imshow(iradon_img, cmap='gray')
    plt.subplot(143)
    plt.title('FBP with shepp-logan filter')
    plt.imshow(iradon_img_, cmap='gray')
    plt.subplot(144)
    plt.title('predicted image')
    plt.imshow(pre, cmap='gray')
    plt.show()
