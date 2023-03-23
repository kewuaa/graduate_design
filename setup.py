import os

import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize


cython_exts = [
    Extension(
        name='src.graduate_design.data.generator.circle',
        sources=[
            'cython\\circle.pyx',
        ]
    ),
    Extension(
        name='src.graduate_design.data.transformer.ctrans',
        sources=[
            'cython\\transform.pyx',
        ],
        include_dirs=[
            np.get_include(),
        ],
    ),
]
opencv_home = os.environ['OPENCV_HOME']
pybind11_home = os.environ['PYBIND11_HOME']
exts = [
    Extension(
        name='src.graduate_design.data.transformer.cpptrans',
        sources=[
            'cpp\\radon_transform\\src\\main.cpp'
        ],
        include_dirs=[
            pybind11_home + '/include',
            opencv_home + '/include'
        ],
        library_dirs=[
            opencv_home + '/x64/mingw/lib'
        ],
        libraries=[
            'opencv_world470',
        ]
    )
]
setup(
    ext_modules=cythonize(cython_exts) + exts
)
