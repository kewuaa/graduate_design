from pathlib import Path
import asyncio

import cv2


class Transformer:
    def __init__(
        self,
        img_num: int,
        start_angle: int,
        end_angle: int,
        theta_step: float,
        data_path: Path,
    ) -> None:
        self._img_num = img_num
        self._theta_step = theta_step
        self._start_angle = start_angle
        self._end_angle = end_angle
        self._loop = asyncio.get_event_loop()
        self._source_path = data_path / 'imgs'
        self._target_path = data_path / 'transformed_imgs'
        self._target_path.mkdir(parents=True, exist_ok=True)

    def _radon(self, img):
        img = cv2.normalize(
            img, None,
            -0.5, 0.5,
            cv2.NORM_MINMAX,
            cv2.CV_32F
        )
        return cv2.ximgproc.RadonTransform(
            img,
            theta=self._theta_step,
            crop=True,
            start_angle=self._start_angle,
            end_angle=self._end_angle,
            norm=True,
        )

    async def _transform(self, name, refresh=None):
        img_file = self._source_path / name
        save_path = self._target_path / name
        while not img_file.exists():
            await asyncio.sleep(1.5)
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
