# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language=c++
import cv2
import numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from .radon_transform cimport RadonTransformer


# cdef extern from "radon_transform.hpp" nogil:
#     cdef cppclass RadonTransformer:
#         RadonTransformer() except +
#         RadonTransformer(
#             float theta,
#             float start_angle,
#             float end_angle,
#             bint crop,
#             bint norm,
#             bint add_noise
#         ) except +
#         void radon_transform_with_noise(const char* _bytes, unsigned int byte_length, vector[unsigned char]& out_buf)


cdef class Radon:
    cdef RadonTransformer radon_transformer

    def __init__(
        self,
        float theta,
        float start_angle,
        float end_angle,
        bint crop,
        bint norm,
        bint add_noise
    ):
        self.radon_transformer = RadonTransformer(
            theta,
            start_angle,
            end_angle,
            crop, norm,
            add_noise
        )

    cpdef bytes run(self, bytes data):
        cdef vector[unsigned char] out_buf
        cdef const char* _bytes = data
        cdef unsigned int _bytes_length = len(data)
        cdef string buf
        with nogil:
            self.radon_transformer.radon_transform_with_noise(
                _bytes,
                _bytes_length,
                out_buf
            )
            buf = string(<const char*>out_buf.data(), out_buf.size())
        return buf
