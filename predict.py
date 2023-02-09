from pathlib import Path

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import cv2

from project import model
from project.data import Dataset
image_dir = Path('./data/transformed_imgs')
label_dir = Path('./data/imgs')


if __name__ == "__main__":
    net = model.Automap()
    net.load('./checkpoints/checkpoint_epoch_1.pth')
    with Dataset() as dataset:
        loader = DataLoader(dataset, batch_size=1)
        for image, label in loader:
            pre = net(label)
            break

    ori_train = image
    img_result_train = np.squeeze(pre)
    reconstruction_fbp_train = label

    # plt.subplot(231), plt.imshow(ori_test, cmap='gray')
    # plt.title('Original test set'), plt.xticks([]), plt.yticks([])

    # plt.subplot(232), plt.imshow(img_result_test, cmap='gray')
    # plt.title('Recon'), plt.xticks([]), plt.yticks([])

    # plt.subplot(233), plt.imshow(reconstruction_fbp, cmap='gray')
    # plt.title('Recon by iradon'), plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(ori_train, cmap='gray')
    plt.title('Original train set'), plt.xticks([]), plt.yticks([])

    plt.subplot(235), plt.imshow(img_result_train, cmap='gray')
    plt.title('Recon'), plt.xticks([]), plt.yticks([])

    plt.subplot(236), plt.imshow(reconstruction_fbp_train, cmap='gray')
    plt.title('Recon by iradon'), plt.xticks([]), plt.yticks([])

    plt.show()
