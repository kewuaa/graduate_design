# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language=c
from libc.stdlib cimport rand, srand
from libc.time cimport time
import cv2
import numpy as np
cimport numpy as cnp
srand(<unsigned int> time(NULL))


cpdef cnp.ndarray[double, ndim=2] radon(
        cnp.ndarray[unsigned char, ndim=2] img,
        double theta_step
    ):
    cdef:
        unsigned int i, j
        unsigned int projection_num
        double angle
        int[2] center
        unsigned char[:, ::1] img_view
        cnp.ndarray[unsigned char, ndim=2] img_rotated
        cnp.ndarray[double, ndim=2] sinogram
    img_view = img
    center[0] = img_view.shape[0] // 2
    center[1] = img_view.shape[1] // 2
    projection_num = <int>(180 / theta_step)
    sinogram = np.empty((img_view.shape[0], projection_num))
    i = 0
    angle = 0.
    while i < projection_num:
        img_rotated = cv2.warpAffine(
            img,
            cv2.getRotationMatrix2D(center, -angle, 1.),
            (img_view.shape[0], img_view.shape[1]),
            None,
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
            borderValue=255
        )
        if i % ((rand() % 3 + 1) + 3) == 0:
        # if i % random.randint(3, 6) == 0:
            sinogram[:, i] = np.convolve(
                    img_rotated.astype(np.float64).sum(axis=0),
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    mode='same'
                    )
            # for j in range(img_view.shape[0]):
            #     img_rotated[j, ...] = np.convolve(
            #         img_rotated[j, ...],
            #         [1, 1, 1, 1, 1, 1, 1, 1, 1],
            #         mode='same'
            #     )
        else:
            sinogram[:, i] = img_rotated.astype(np.float64).sum(axis=0)
        i += 1
        angle += theta_step
    return (sinogram - sinogram.min()) / sinogram.ptp() * 255
