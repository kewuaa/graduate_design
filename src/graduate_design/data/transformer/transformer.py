import os
import asyncio
from distutils import spawn
from pathlib import Path
from functools import partial

import cv2

opencv_home = os.environ.get('OPENCV_HOME')
if opencv_home is None:
    raise RuntimeError('OpenCV library not find')
mingw_path = spawn.find_executable('gcc')
if not mingw_path:
    raise RuntimeError('stdc lib not find')
os.add_dll_directory(mingw_path + '/..')
os.add_dll_directory(opencv_home + '/x64/mingw/bin')
from . import cpptrans


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
        self._radon = partial(
            cpptrans.radon_transform_with_noise,
            theta=theta_step,
            start_angle=start_angle,
            end_angle=end_angle,
            add_noise=add_noise
        )

    # def _radon(self, img):
    #     img = cv2.normalize(
    #         img, None,
    #         -0.5, 0.5,
    #         cv2.NORM_MINMAX,
    #         cv2.CV_32F
    #     )
    #     return cv2.ximgproc.RadonTransform(
    #         img,
    #         theta=self._theta_step,
    #         crop=True,
    #         start_angle=self._start_angle,
    #         end_angle=self._end_angle,
    #         norm=True,
    #     )

    async def _transform(self, name, refresh=None):
        img_file = self._source_path / name
        save_path = self._target_path / name
        while not img_file.exists():
            await asyncio.sleep(1.5)
        # image = None
        # while image is None:
        #     image = await self._loop.run_in_executor(
        #         None,
        #         cv2.imread,
        #         str(img_file),
        #         cv2.IMREAD_GRAYSCALE
        #     )
        # sinogram = self._radon(image)
        sinogram = await self._loop.run_in_executor(
            None,
            self._radon,
            str(img_file),
        )
        await self._loop.run_in_executor(
            None,
            cv2.imwrite,
            str(save_path),
            sinogram
        )
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
