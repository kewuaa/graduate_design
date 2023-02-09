from pathlib import Path

from matplotlib import pyplot as plt
import cv2

from project import model
image_dir = Path('./data/transformed_imgs')
label_dir = Path('./data/imgs')


if __name__ == "__main__":
    img_n = 1
    net = model.Automap()
    net.load('./checkpoints/checkpoint_epoch_1.pth')
    image = image_dir / f'{img_n}.png'
    label = label_dir / f'{img_n}.png'
    image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(str(label), cv2.IMREAD_GRAYSCALE)
    image = cv2.normalize(image, None, -0.5, 0.5, cv2.NORM_MINMAX, cv2.CV_32F)
    label = cv2.normalize(label, None, -0.5, 0.5, cv2.NORM_MINMAX, cv2.CV_32F)
    pre = model(image)

    ori_train = image
    img_result_train = pre
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
