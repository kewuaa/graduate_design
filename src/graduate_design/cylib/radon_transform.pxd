from libcpp.vector cimport vector


cdef extern from "radon_transform.hpp" namespace "algorithm" nogil:
    cdef cppclass RadonTransformer:
        RadonTransformer() except +
        RadonTransformer(
            float theta,
            float start_angle,
            float end_angle,
            bint crop,
            bint norm,
            bint add_noise
        ) except +
        void radon_transform_with_noise(
            const unsigned char* _bytes,
            unsigned int byte_length,
            vector[unsigned char]& out_buf
        )
