from typing import Callable
from functools import wraps
from time import time
from pathlib import Path


def find_root(start_path: str, rootmarks: list) -> str:
    def search_marks(path: Path) -> bool:
        for mark in rootmarks:
            if list(path.glob(mark)):
                return True
        return False
    start_path = Path(start_path).resolve()
    if not start_path.exists():
        raise RuntimeError('start path not exists')
    if search_marks(start_path):
        return start_path
    for _dir in start_path.parents:
        if search_marks(_dir):
            return str(_dir)


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f'function {func.__name__} costs {end - start} ms')
        return result
    return wrapped_func
