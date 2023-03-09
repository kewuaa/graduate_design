import random

import cv2
import numpy as np


def radon(img: np.ndarray, theta_step: float) -> np.ndarray:
    shape = img.shape
    center = [size // 2 for size in shape]
    projection_num = int(180 / theta_step)
    sinogram = np.empty([shape[0], projection_num])
    for i, angle in enumerate(np.arange(0, 180, theta_step)):
        M = cv2.getRotationMatrix2D(center, -angle, 1.)
        img_rotated = cv2.warpAffine(
            img,
            M,
            shape,
            None,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
            borderValue=255
        )
        # if i % random.randint(3, 6) == 0:
        #     for j in range(shape[0]):
        #         img_rotated[j, ...] = np.convolve(
        #             img_rotated[j, ...],
        #             np.ones((10, )),
        #             mode='same'
        #         )
        sinogram_ = img_rotated.astype(np.float64).sum(axis=0)
        sinogram[:, i] = sinogram_
    sinogram = cv2.normalize(sinogram, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return sinogram
