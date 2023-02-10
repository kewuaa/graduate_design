# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from libc.stdlib cimport rand, srand
from libc.time cimport time
from libc.math cimport pow as c_pow, sqrt as c_sqrt, fmin as c_min


cdef inline bint overlap(double *c1, double *c2) nogil:
    cdef:
        double c1_size, c2_size, c1_x, c2_x, c1_y, c2_y
        bint x_overlap, y_overlap
    c1_size = c1[2] + 1.0
    c2_size = c2[2] + 1.0
    c1_x, c1_y = c1[0], c1[1]
    c2_x, c2_y = c2[0], c2[1]
    x_overlap = c1_x - c1_size <= c2_x + c2_size and c1_x + c1_size >= c2_x - c2_size
    y_overlap = c1_y - c1_size <= c2_y + c2_size and c1_y + c1_size >= c2_y - c2_size
    return x_overlap and y_overlap


cdef class Circle:
    cdef:
        double left, bottom, width, height, center[2]
        double size_range, min_size, inscribed_r
    def __init__(
        self,
        double left,
        double right,
        double bottom,
        double top,
        double min_size,
        double max_size
    ):
        self.left = left
        self.width = right - left
        self.bottom = bottom
        self.height = top - bottom
        self.center[0] = left + self.width / 2
        self.center[1] = bottom + self.height / 2
        self.min_size = min_size
        self.size_range = max_size - min_size
        self.inscribed_r = c_min(self.width, self.height) / 2

        srand(<unsigned int> time(NULL))

    cdef inline void generate_one(self, double[::1] circle) nogil:
        cdef:
            double size, padded_size, x, y
        size = (rand() % self.size_range + self.min_size) \
            if self.size_range != 0 else self.min_size
        padded_size = size + 2.0
        x = rand() % (self.width - 2 * padded_size) + self.left + padded_size
        y = rand() % (self.height - 2 * padded_size) + self.bottom + padded_size

        cdef double dist = c_sqrt(c_pow(x - self.center[0], 2) + c_pow(y - self.center[1], 2))
        if dist + padded_size > self.inscribed_r:
            self.generate_one(circle)
        else:
            circle[0] = x
            circle[1] = y
            circle[2] = size

    cpdef list generate(self, int max_num):
        cdef:
            double circles[9][3]
            double[:, ::1] circles_view = circles
            double circle[3]
            double[::1] circle_view = circle
            unsigned int i, num = 0, cycle_num = 0
        with nogil:
            while num < max_num and cycle_num < 1000:
                self.generate_one(circle_view)
                for i in range(num):
                    if overlap(circles[i], circle):
                        cycle_num += 1
                        break
                else:
                    circles_view[num][:] = circle_view
                    num += 1
        cdef:
            list es = []
            tuple c
        for i in range(num):
            c = (
                (circles_view[i][0] - circles_view[i][2], circles_view[i][1] - circles_view[i][2]),
                (circles_view[i][0] + circles_view[i][2], circles_view[i][1] + circles_view[i][2]),
            )
            es.append(c)
        return es
