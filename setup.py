import os

from setuptools import setup, Extension
opencv_home = os.environ['OPENCV_HOME']
files = (
    'src\\graduate_design\\cylib\\circle.c',
    'src\\graduate_design\\cylib\\transform.cpp'
)
suffix = ''
if not all(os.path.exists(file) for file in files):
    from Cython.Build import cythonize
    suffix = 'pyx'


exts = [
    Extension(
        name='graduate_design.cylib.circle',
        sources=[
            f'src\\graduate_design\\cylib\\circle.{suffix or "c"}',
        ]
    ),
    Extension(
        name='graduate_design.cylib.ctrans',
        sources=[
            f'src\\graduate_design\\cylib\\transform.{suffix or "cpp"}',
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
if suffix:
    exts = cythonize(exts)
setup(
    ext_modules=exts,
    zip_safe=False,
    package_dir={'graduate_design': 'src/graduate_design'}
)
