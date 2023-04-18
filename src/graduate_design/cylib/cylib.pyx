# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language=c++
# distutils: include_dirs=src/graduate_design/cylib/cpp/include
# distutils: sources=src/graduate_design/cylib/cpp/src/graph.cpp src/graduate_design/cylib/cpp/src/radon_transform.cpp
from io import BytesIO
from PIL import Image, ImageDraw

from libc.stdlib cimport rand, srand
from libc.time cimport time
from libcpp.vector cimport vector
from cpython cimport array
import array
from .graph cimport Generator, Area
from .radon_transform cimport RadonTransformer


ctypedef unsigned char uchar
ctypedef enum GraphType:
    ALL_ELLIPSE
    ALL_POLYGON
    RANDOM


ctypedef fused Pixel:
    uchar
    uchar[::1]


cdef class Graph:

    cdef Generator generator
    cdef unsigned short img_size

    def __init__(
        self,
        unsigned short img_size,
        tuple radius
    ):
        self.img_size = img_size
        if len(radius) > 1:
            self.generator = Generator(img_size, radius[0], radius[1])
        else:
            self.generator = Generator(img_size, radius[0])
        srand(<unsigned int>time(NULL))

    cpdef bytes gen(
        self,
        unsigned short num,
        Pixel pixel,
        GraphType config
    ):
        cdef vector[Area] areas
        cdef unsigned short n
        with nogil:
            areas.reserve(num)
            n = self.generator.gen(num, areas.data())
        cdef uchar alpha
        cdef array.array points
        cdef unsigned short i
        cdef unsigned short n_sides
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        for i in range(n):
            if Pixel is uchar:
                alpha = pixel
            else:
                alpha = (
                    rand() % ((pixel[1] - pixel[0]) / pixel[2])
                ) * pixel[2] + pixel[0]
            n_sides = rand() % 8
            if config == GraphType.ALL_ELLIPSE or (
                    config == GraphType.RANDOM and n_sides < 3):
                points = array.array('f', [0.] * 4)
                self.generator.gen_circle(points.data.as_floats, areas[i])
                draw.ellipse(points, outline=alpha, fill=alpha)
            else:
                if n_sides < 3:
                    n_sides = rand() % 5 + 3
                if rand() % 2:
                    draw.regular_polygon(
                        (areas[i].center, areas[i].radius), n_sides,
                        rotation=rand() % 360, fill=alpha, outline=alpha,
                    )
                else:
                    points = array.array('f', [0.] * n_sides * 2)
                    self.generator.gen_polygon(
                        points.data.as_floats,
                        areas[i],
                        n_sides
                    )
                    draw.polygon(points, outline=alpha, fill=alpha)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()


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

    cpdef bytes run(self, const uchar[::1] data):
        cdef vector[uchar] out_buf
        with nogil:
            self.radon_transformer.radon_transform_with_noise(
                &data[0],
                data.shape[0],
                out_buf
            )
        return out_buf.data()[:out_buf.size()]
