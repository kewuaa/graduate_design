from matplotlib import pyplot as plt
from skimage.transform import iradon

from project import model


if __name__ == "__main__":
    # net = model.Automap()
    net = model.UNet(3)
    net.load('./checkpoints/checkpoint_unet_epoch_5.pth')
    img, label, pre = net.predict(3)
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.subplot(132)
    plt.imshow(label, cmap='gray')
    plt.subplot(133)
    plt.imshow(pre, cmap='gray')
    plt.show()
