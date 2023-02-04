from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    name='circle',
    sources=['./circle.pyx'],
)
setup(ext_modules=cythonize(ext))
