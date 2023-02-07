from pathlib import Path
import asyncio

import cv2

from ...logging import logger


class Transformer:
    def __init__(
        self,
        img_num: int,
        theta_step: float,
        start_angle: int = 0,
        end_angle: int = 180
    ) -> None:
        self._img_num = img_num
        self._theta_step = theta_step
        self._start_angle = start_angle
        self._end_angle = end_angle
        self._loop = asyncio.get_event_loop()
        self._source_path = Path('./project/data/imgs')
        self._target_path = Path('./project/data/transformed_imgs')
        self._target_path.mkdir(exist_ok=True)

    def _radon(self, img):
        return cv2.ximgproc.RadonTransform(
            img,
            theta=self._theta_step,
            crop=True,
            start_angle=self._start_angle,
            end_angle=self._end_angle,
            norm=True,
        )

    async def _transform(self, name):
        img_file = self._source_path / name
        save_path = self._target_path / name
        while not img_file.exists():
            await asyncio.sleep(0.5)
        image = None
        while image is None:
            image = await self._loop.run_in_executor(
                None,
                cv2.imread,
                str(img_file),
                cv2.IMREAD_GRAYSCALE
            )
        # image = np.where(mask, image, 0)
        sinogram = self._radon(image)
        await self._loop.run_in_executor(
            None,
            cv2.imwrite,
            str(save_path),
            sinogram
        )

    async def transform(self) -> None:
        batch_size = 10
        logger.info('making radon transform......')
        for i in range(0, self._img_num, batch_size):
            tasks = [
                asyncio.create_task(self._transform(f'{i + j + 1}.png'))
                for j in range(batch_size)
            ]
            for task in tasks:
                await task
        logger.info('transform done')
