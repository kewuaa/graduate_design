from pathlib import Path
import os

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class Build(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == 'mingw32':
            for ext in self.extensions:
                ext.extra_link_args = [
                    '-static-libgcc',
                    '-static-libstdc++',
                    '-Wl,-Bstatic,--whole-archive',
                    '-lwinpthread',
                    '-Wl,--no-whole-archive',
                ]
        super().build_extensions()



lib_type = 'static'
opencv_home = f"{os.environ['OPENCV_HOME']}/{lib_type}"
opencv_include_dir = opencv_home + '/include'
opencv_lib_dir = opencv_home + '/x64/mingw/lib'
library_dirs = [opencv_lib_dir]
libraries = ['opencv_world470']
if lib_type == 'static':
    libraries += list(
        f.stem[3:]
        for f in Path(opencv_lib_dir).iterdir()
        if f.suffix == '.a'
    )
suffix = 'pyx'


exts = [
    Extension(
        name='graduate_design.cylib.cylib',
        sources=[
            f'src\\graduate_design\\cylib\\cylib.{suffix or "cpp"}',
        ],
        include_dirs=[
            opencv_include_dir,
            'src\\graduate_design\\cylib\\cpp\\include'
        ],
        library_dirs=library_dirs,
        libraries=libraries
    ),
]
if suffix:
    from Cython.Build import cythonize
    exts = cythonize(exts)
setup(
    ext_modules=exts,
    zip_safe=False,
    package_dir={'graduate_design': 'src/graduate_design'},
    cmdclass={'build_ext': Build}
)
