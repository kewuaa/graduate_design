import os

from setuptools import setup, Extension
from Cython.Build import cythonize
opencv_home = os.environ['OPENCV_HOME']


cython_exts = [
    Extension(
        name='src.graduate_design.cylib.circle',
        sources=[
            'src\\graduate_design\\cylib\\circle.pyx',
        ]
    ),
    Extension(
        name='src.graduate_design.cylib.ctrans',
        sources=[
            'src\\graduate_design\\cylib\\transform.pyx',
            'src\\graduate_design\\cylib\\cpp\\src\\radon_transform.cpp'
        ],
        include_dirs=[
            opencv_home + '/include',
            'src\\graduate_design\\cylib\\cpp\\include'
        ],
        library_dirs=[
            opencv_home + '/x64/mingw/lib'
        ],
        libraries=[
            'opencv_world470',
        ]
    ),
]
setup(
    ext_modules=cythonize(cython_exts)
)
