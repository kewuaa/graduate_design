from typing import Union
from functools import partial
from pathlib import Path
import random
import array
import asyncio

import aiofiles

from ...cylib import Graph


class Generator:
    def __init__(
        self,
        img_num: int,
        img_size: int,
        pixel: Union[list, int],
        graph_num: Union[list, int],
        graph_size: Union[list, int],
        data_path: Path,
    ) -> None:
        self._img_num = img_num
        self._img_size = img_size
        if type(graph_num) is int:
            self._get_circle_num = lambda: graph_num
        else:
            self._get_circle_num = partial(random.randint, *graph_num)
        self._graph = Graph(
            img_size,
            (graph_size,) if type(graph_size) is int
            else tuple(graph_size)
        )
        self._pixel = pixel if type(pixel) is int else array.array('B', pixel)
        self._loop = asyncio.get_event_loop()
        self._img_save_path = data_path / 'imgs'
        self._img_save_path.mkdir(parents=True, exist_ok=True)

    async def _generate_one(self, index: int, refresh=None) -> None:
        img_bytes = await self._loop.run_in_executor(
            None,
            self._graph.gen,
            self._get_circle_num(),
            self._pixel
        )
        async with aiofiles.open(
            self._img_save_path / f'{index + 1}.png', 'wb'
        ) as f:
            await f.write(img_bytes)
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
