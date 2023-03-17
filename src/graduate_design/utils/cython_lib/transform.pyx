# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import cv2
import numpy as np
cimport numpy as cnp


cpdef cnp.ndarray[double, ndim=2] radon(
        cnp.ndarray[unsigned char, ndim=2] img,
        double theta_step
    ):
    cdef:
        unsigned int i
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
        sinogram[:, i] = img_rotated.astype(np.float64).sum(axis=0)
        i += 1
        angle += theta_step
    return (sinogram - sinogram.min()) / sinogram.ptp() * 255
