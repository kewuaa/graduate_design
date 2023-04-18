import os

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class Build(build_ext):
    def build_extensions(self):
        if 'zig' in self.compiler.cc:
            self.compiler.dll_libraries = []
            self.compiler.set_executable(
                'compiler_so',
                f'{self.compiler.cc} -O -Wall -lc++'
            )
            for ext in self.extensions:
                ext.undef_macros = ['_DEBUG']
        super().build_extensions()



suffix = 'pyx'
include_dirs = [os.environ['INCLUDE']]
library_dirs = [os.environ['LIB']]
libraries = ['opencv_world460']
exts = [
    Extension(
        name='graduate_design.cylib.cylib',
        sources=[
            f'src\\graduate_design\\cylib\\cylib.{suffix}',
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
    ),
]
if suffix == 'pyx':
    from Cython.Build import cythonize
    exts = cythonize(exts)
setup(
    ext_modules=exts,
    zip_safe=False,
    package_dir={'graduate_design': 'src/graduate_design'},
    cmdclass={'build_ext': Build}
)
