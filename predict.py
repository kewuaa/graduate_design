from matplotlib import pyplot as plt
import cv2

from project import model


if __name__ == "__main__":
    # net = model.Automap()
    # net.load('./checkpoints/checkpoint_automap_epoch_5.pth')
    net = model.UNet(3)
    net.load('./checkpoints/checkpoint_unet_epoch_5.pth')
    img, label, pre = net.validate(3)
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.subplot(132)
    plt.imshow(label, cmap='gray')
    plt.subplot(133)
    plt.imshow(pre, cmap='gray')
    plt.show()
    img = r
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    pre = net.predict(img)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(pre, cmap='gray')
    plt.show()
