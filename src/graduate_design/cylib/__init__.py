# from distutils import spawn
# import os
# opencv_home = os.environ.get('OPENCV_HOME')
# if opencv_home is None:
#     raise RuntimeError('OpenCV library not find')
# mingw_path = spawn.find_executable('gcc')
# if not mingw_path:
#     raise RuntimeError('stdc lib not find')
# os.add_dll_directory(mingw_path + '/..')
# os.add_dll_directory(opencv_home + '/x64/mingw/bin')
from .cylib import Graph, Radon
