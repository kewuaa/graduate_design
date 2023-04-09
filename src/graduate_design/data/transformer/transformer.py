import os
import asyncio
from distutils import spawn
from pathlib import Path

import aiofiles

opencv_home = os.environ.get('OPENCV_HOME')
if opencv_home is None:
    raise RuntimeError('OpenCV library not find')
mingw_path = spawn.find_executable('gcc')
if not mingw_path:
    raise RuntimeError('stdc lib not find')
os.add_dll_directory(mingw_path + '/..')
os.add_dll_directory(opencv_home + '/x64/mingw/bin')
from ...cylib import ctrans


class Transformer:
    def __init__(
        self,
        img_num: int,
        start_angle: int,
        end_angle: int,
        theta_step: float,
        data_path: Path,
        add_noise: bool,
    ) -> None:
        self._img_num = img_num
        self._loop = asyncio.get_event_loop()
        self._source_path = data_path / 'imgs'
        self._target_path = data_path / 'transformed_imgs'
        self._target_path.mkdir(parents=True, exist_ok=True)
        self._radon = ctrans.Radon(theta_step, start_angle, end_angle, 1, 1, 1)

    async def _transform(self, name, refresh=None):
        img_file = self._source_path / name
        save_path = self._target_path / name
        while not img_file.exists():
            await asyncio.sleep(1.5)
        async with aiofiles.open(img_file, 'rb') as f:
            data = await f.read()
        sinogram = await self._loop.run_in_executor(
            None,
            self._radon.run,
            data,
        )
        async with aiofiles.open(save_path, 'wb') as f:
            await f.write(sinogram)
        if callable(refresh):
            refresh()

    async def transform(self, refresh=None) -> None:
        batch_size = 10
        for i in range(0, self._img_num, batch_size):
            tasks = [
                asyncio.create_task(
                    self._transform(f'{i + j + 1}.png', refresh)
                )
                for j in range(batch_size)
            ]
            for task in tasks:
                await task
