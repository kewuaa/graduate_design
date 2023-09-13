import os
import sys
from distutils.unixccompiler import UnixCCompiler

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class Build(build_ext):
    def build_extensions(self):
        if isinstance(self.compiler, UnixCCompiler):
            if 'zig' in self.compiler.cc:
                self.compiler.dll_libraries = []
                self.compiler.set_executable(
                    'compiler_so',
                    f'{self.compiler.cc} -O -Wall -lc++'
                )
                for ext in self.extensions:
                    ext.undef_macros = ['_DEBUG']
        super().build_extensions()


if "--use-cython" in sys.argv:
    sys.argv.remove("--use-cython")
    use_cython = True
else:
    use_cython = False
suffix = 'pyx' if use_cython else "cpp"
include_dirs = os.environ['INCLUDE'].rstrip(";").split(";")
include_dirs.append("./src/graduate_design/cylib/cpp/include")
library_dirs = os.environ['LIB'].rstrip(";").split(";")
libraries = ['opencv_world460.dll']
exts = [
    Extension(
        name='graduate_design.cylib.cylib',
        sources=[
            f'src\\graduate_design\\cylib\\cylib.{suffix}',
            "./src/graduate_design/cylib/cpp/src/graph.cpp",
            "./src/graduate_design/cylib/cpp/src/radon_transform.cpp"
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
    ),
]
if use_cython:
    from Cython.Build import cythonize
    exts = cythonize(exts)
setup(
    ext_modules=exts,
    cmdclass={'build_ext': Build}
)
