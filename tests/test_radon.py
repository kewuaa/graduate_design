import cv2
from matplotlib import pyplot as plt

from project.utils.transform import radon


def test_radon():
    img = cv2.imread('./data/imgs/1.png', cv2.IMREAD_GRAYSCALE)
    img = 255 - img
    img = cv2.normalize(img, None, -0.5, +0.5, cv2.NORM_MINMAX, cv2.CV_32F)
    sinogram = radon(img, 1.5)
    plt.imshow(sinogram, cmap='gray')
    plt.show()
