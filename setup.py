import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize


exts = [
    Extension(
        name='src.graduate_design.utils.cython_lib.circle',
        sources=['src\\graduate_design\\utils\\cython_lib\\circle.pyx']
    ),
    Extension(
        name='src.graduate_design.utils.cython_lib.transform',
        sources=['src\\graduate_design\\utils\\cython_lib\\transform.pyx'],
        include_dirs=[np.get_include()]
    )
]
setup(
    ext_modules=cythonize(exts)
)
