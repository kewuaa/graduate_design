import cv2
from matplotlib import pyplot as plt

from src.graduate_design.utils.cython_lib import transform


def test_radon():
    img = cv2.imread('./data/imgs/1.png', cv2.IMREAD_GRAYSCALE)
    # img = cv2.normalize(img, None, -0.5, +0.5, cv2.NORM_MINMAX, cv2.CV_32F)
    sinogram = transform.radon(img, 1.5)
    plt.imshow(sinogram, cmap='gray')
    plt.show()
