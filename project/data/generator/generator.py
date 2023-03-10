from functools import partial
from pathlib import Path
import random
import asyncio

from PIL import Image, ImageDraw

from .cython_lib import circle


class Generator:
    def __init__(
        self,
        img_num: int,
        img_size: int,
        pixel: list,
        circle_num: list,
        circle_size: list,
        data_path: Path,
    ) -> None:
        self._img_num = img_num
        self._img_size = img_size
        if len(set(circle_num)) == 1:
            circle_num = circle_num[0]
            self._get_circle_num = lambda: circle_num
        else:
            self._get_circle_num = partial(random.randint, *circle_num)
        self._Circle = circle.Circle(
            0.,
            img_size,
            0.,
            img_size,
            *circle_size
        )
        self._pixel = tuple(range(*pixel))
        self._loop = asyncio.get_event_loop()
        self._img_save_path = data_path / 'imgs'
        self._img_save_path.mkdir(parents=True, exist_ok=True)

    async def _generate_one(self, index: int, refresh=None) -> None:
        circles = await self._loop.run_in_executor(
            None,
            self._Circle.generate,
            self._get_circle_num(),
        )
        img = Image.new('L', (self._img_size, self._img_size), 255)
        draw = ImageDraw.Draw(img)
        for c in circles:
            alpha = random.choice(self._pixel)
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
        if callable(refresh):
            refresh()

    async def generate(self, refresh=None) -> None:
        batch_size = 10
        for i in range(0, self._img_num, batch_size):
            tasks = [
                self._loop.create_task(self._generate_one(i + j, refresh))
                for j in range(batch_size)
            ]
            for task in tasks:
                await task
