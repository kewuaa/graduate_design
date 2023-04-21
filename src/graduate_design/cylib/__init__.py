os = __import__('os')
os.add_dll_directory(os.environ['DLL'])
from .cylib import Graph, Radon, Ring
