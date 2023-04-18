

cdef extern from "graph.hpp" namespace "graph" nogil:
    ctypedef struct Area:
        unsigned short radius
        float center[2]

    cdef cppclass Generator:
        Generator() except +
        Generator(unsigned int img_size, unsigned int radius) except +
        Generator(
            unsigned int img_size,
            unsigned int min_radius,
            unsigned int max_radius
        ) except +
        void gen_circle(float* points, const Area& area)
        void gen_polygon(float* points, const Area& area, unsigned short n_sides)
        unsigned short gen(unsigned short num, Area* areas)
