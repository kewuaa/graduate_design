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
