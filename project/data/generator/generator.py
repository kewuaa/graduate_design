from functools import partial
from pathlib import Path
import random
import asyncio

from PIL import Image, ImageDraw

from .cython_lib import circle
from ...logging import logger


class Generator:
    def __init__(
        self,
        img_num: int,
        img_size: int,
        max_circle_num: int,
        min_circle_size: int,
        max_circle_size: int
    ) -> None:
        self._img_num = img_num
        self._img_size = img_size
        self._max_circle_num = max_circle_num
        self._Circle = circle.Circle(
            0.,
            img_size,
            0.,
            img_size,
            min_circle_size,
            max_circle_size
        )
        self._pixel_option = list(range(0, 128, 10))
        self._loop = asyncio.get_event_loop()
        self._img_save_path = Path('./project/data/imgs')
        self._img_save_path.mkdir(exist_ok=True)

    async def _generate_one(self, index: int) -> None:
        circles = await self._loop.run_in_executor(
            None,
            self._Circle.generate,
            random.randint(1, self._max_circle_num),
        )
        img = Image.new('L', (self._img_size, self._img_size), 255)
        draw = ImageDraw.Draw(img)
        for c in circles:
            alpha = random.choice(self._pixel_option)
            left_top, right_bottom = c
            draw.ellipse(
                (left_top, right_bottom),
                fill=alpha,
                outline=alpha,
            )
        await self._loop.run_in_executor(
            None,
            partial(
                img.save,
                str(self._img_save_path / f'{index + 1}.png'),
                dpi=(300, 300),
            )
        )

    async def generate(self) -> None:
        batch_size = 10
        logger.info('generating image......')
        for i in range(0, self._img_num, batch_size):
            tasks = [
                self._loop.create_task(self._generate_one(i + j))
                for j in range(batch_size)
            ]
            for task in tasks:
                await task
        logger.info('generate done')
