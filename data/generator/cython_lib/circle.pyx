# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
from libc.stdlib cimport rand, srand
from libc.time cimport time
from libc.math cimport pow as c_pow, sqrt as c_sqrt
cdef:
    double region_size = 140.0
    double max_size = 30
    double min_size = 10
    double left_bottom[2]
left_bottom[0] = 0.0
left_bottom[1] = 0.0
srand(<unsigned int>time(NULL))


cdef inline bint outside(double x, double y, double size) nogil:
    cdef double dist = c_sqrt(c_pow(x - 70, 2) + c_pow(y - 70, 2))
    return dist + size > 70.0


cdef inline void generate_circle(double[::1] circle) nogil:
        cdef:
            double size, size_pad, x, y
        size = rand() % (max_size - min_size) + min_size
        size_pad = size + 2.0
        x = rand() % (region_size - 2 * size_pad) + left_bottom[0] + size_pad
        y = rand() % (region_size - 2 * size_pad) + left_bottom[1] + size_pad
        if outside(x, y, size_pad):
            generate_circle(circle)
        else:
            circle[0] = x
            circle[1] = y
            circle[2] = size


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


cpdef list generate(int max_num):
    cdef:
        double circles[3][3]
        double[:, ::1] circles_view = circles
        double circle[3]
        double[::1] circle_view = circle
        unsigned int i, num = 0, cycle_num = 0
    with nogil:
        while num < max_num and cycle_num < 1000:
            generate_circle(circle_view)
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
        # c = (circles[i][0], circles[i][1], circles[i][2])
        es.append(c)
    return es
