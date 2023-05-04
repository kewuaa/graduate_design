import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import iradon, iradon_sart
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
iradon_sart = timer(iradon_sart)
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
    img = img / 255
    label = label / 255
    pre = pre / 255
    thetas = np.arange(0., 180., step=config.theta_step)
    iradon_img = iradon(img, theta=thetas, filter_name='ramp')
    sart_img = iradon_sart(img, theta=thetas)
    # iradon_img_ = iradon(img, theta=thetas, filter_name='shepp-logan')
    iradon_img = (iradon_img - iradon_img.min()) / iradon_img.ptp()
    sart_img = (sart_img - sart_img.min()) / sart_img.ptp()
    # iradon_img_ = (iradon_img_ - iradon_img_.min()) / iradon_img_.ptp()
    mse_net = MSE(label, pre)
    psnr_net = PSNR(label, pre)
    ssim_net = SSIM(label, pre)
    mse_fbp = MSE(label, iradon_img)
    psnr_fbp = PSNR(label, iradon_img)
    ssim_fbp = SSIM(label, iradon_img)
    mse_sart = MSE(label, sart_img)
    psnr_sart = PSNR(label, sart_img)
    ssim_sart = SSIM(label, sart_img)
    # mse_fbp_ = MSE(label, iradon_img_)
    # psnr_fbp_ = PSNR(label, iradon_img_)
    # ssim_fbp_ = SSIM(label, iradon_img_)
    print(f'''
    MSE of FBP is {mse_fbp}
    PSNR of FBP is {psnr_fbp}
    SSIM of FBP is {ssim_fbp}
    MSE of sart is {mse_sart}
    PSNR of sart is {psnr_sart}
    SSIM of sart is {ssim_sart}
    MSE of net is {mse_net}
    PSNR of net is {psnr_net}
    SSIM of net is {ssim_net}
    ''')
    plt.subplot(151)
    plt.title('sinogram')
    plt.imshow(img, cmap='gray')
    axis = plt.gca()
    axis.set_aspect(1. / axis.get_data_ratio())
    plt.subplot(152)
    plt.title('Ground truth')
    plt.imshow(label, cmap='gray')
    plt.subplot(153)
    plt.title('FBP')
    plt.imshow(iradon_img, cmap='gray')
    plt.subplot(154)
    plt.title('sart')
    plt.imshow(sart_img, cmap='gray')
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
    sart_img = iradon_sart(img, theta=thetas)
    plt.subplot(141)
    plt.title('sinogram')
    plt.imshow(img, cmap='gray')
    axis = plt.gca()
    axis.set_aspect(1. / axis.get_data_ratio())
    plt.subplot(142)
    plt.title('FBP')
    plt.imshow(iradon_img, cmap='gray')
    plt.subplot(143)
    plt.title('sart')
    plt.imshow(sart_img, cmap='gray')
    plt.subplot(144)
    plt.title('predicted image')
    plt.imshow(pre, cmap='gray')
    plt.show()
