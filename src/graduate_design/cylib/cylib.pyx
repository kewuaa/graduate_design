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
ctypedef unsigned short uint16


ctypedef fused GraphType:
    uint16
    tuple


ctypedef fused Pixel:
    uchar
    uchar[::1]


cdef class Graph:

    cdef Generator generator
    cdef uint16 img_size

    def __init__(
        self,
        uint16 img_size,
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
        uint16 num,
        Pixel pixel,
        GraphType graph_types
    ):
        cdef vector[Area] areas
        cdef uint16 n
        with nogil:
            areas.reserve(num)
            n = self.generator.gen(num, areas.data())
        cdef uchar alpha
        cdef uint16 i, _type
        cdef array.array points
        img = Image.new('L', (self.img_size, self.img_size), 0)
        draw = ImageDraw.Draw(img)
        for i in range(n):
            if Pixel is uchar:
                alpha = pixel
            else:
                alpha = (
                    rand() % ((pixel[1] - pixel[0]) / pixel[2])
                ) * pixel[2] + pixel[0]
            if GraphType is uint16:
                _type = graph_types
            else:
                _type = graph_types[rand() % len(graph_types)]

            if _type < 3:
                points = array.array('f', [0.] * 4)
                self.generator.gen_circle(points.data.as_floats, areas[i])
                draw.ellipse(points, outline=alpha, fill=alpha)
            elif _type > 100:
                points = array.array('f', [0.] * 4)
                self.generator.gen_circle(points.data.as_floats, areas[i])
                draw.ellipse(
                    points,
                    fill=None,
                    outline=alpha,
                    width=_type - 100
                )
            else:
                draw.regular_polygon(
                    (areas[i].center, areas[i].radius), _type,
                    rotation=rand() % 360, fill=alpha, outline=alpha,
                )
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
