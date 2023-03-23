import asyncio
import random
import threading
from functools import lru_cache, partial
from pathlib import Path

import cv2
import numpy as np
from rich.progress import Progress
from torch import Tensor
from torch.utils.data import Dataset

from .. import config
from ..logging import logger
from .generator import Generator
from .transformer import Transformer
data_path = Path('./data')


def init() -> int:
    async def main():
        progress = Progress()
        generate_task = progress.add_task(
            '[blue]generating',
            total=img_num
        )
        transform_task = progress.add_task(
            '[yellow]radon transformimg',
            total=img_num
        )
        with progress:
            gtask = loop.create_task(generator.generate(
                refresh=partial(progress.update, generate_task, advance=1)
            ))
            ttask = loop.create_task(transformer.transform(
                refresh=partial(progress.update, transform_task, advance=1)
            ))
            await gtask
            await ttask
    config_for_data = config.get('data')
    img_num = config_for_data.image_num
    img_size = config_for_data.image_size
    pixel = config_for_data.pixel
    circle_num = config_for_data.circle_num
    circle_size = config_for_data.circle_size
    angle = config_for_data.angle
    theta_step = config_for_data.theta_step
    logger.info(f'''
    config for data:
        img_size    : {img_size}x{img_size}
        img_num     : {img_num}
        pixel       : {pixel}
        circle_size : {circle_size}
        circle_num  : {circle_num}
        angle       : {angle}
        theta_step  : {theta_step}
    ''')
    global data_path
    data_path = data_path / f'{img_size}x{img_size}_{img_num}_{pixel}_{circle_size}_{circle_num}_{angle}_{theta_step}'
    data_path_exist = data_path.exists()
    if not data_path_exist:
        logger.info('not find data in path, initialize it...')
        generator = Generator(
            img_num,
            img_size,
            pixel,
            circle_num,
            circle_size,
            data_path,
        )
        transformer = Transformer(
            img_num,
            *angle,
            theta_step,
            data_path,
        )
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
        logger.info('data successfully initialized')
    return img_num, img_size, angle, theta_step


class Dataset(Dataset):
    def __init__(self, batch_size: int, pre_process=None):
        self._refresh = None
        self._data = set()
        self._length, self.img_size, self.angle, self.theta_step = init()
        self._img_dir = data_path / 'transformed_imgs'
        self._label_dir = data_path / 'imgs'
        self._loop = asyncio.new_event_loop()
        self._batch_size = batch_size
        self._pre_process = pre_process

    def add_refresh(self, refresh) -> None:
        if not callable(refresh):
            raise RuntimeError('a callable object is needed')
        self._refresh = refresh

    def __enter__(self):
        self._thread = threading.Thread(target=self._start_thread, args=())
        self._thread.start()
        self._load_data()
        return self

    def __exit__(self, *args, **kwargs):
        self._kill_thread()

    def _kill_thread(self):
        def callback():
            self._loop.call_later(0.5, self._loop.stop)
        self._loop.call_soon_threadsafe(callback)
        self._thread.join()

    def _start_thread(self):
        async def _shutdown(loop: asyncio.base_events.BaseEventLoop) -> None:
            """关闭所有待完成的任务."""

            shutdown_tasks = loop.create_task(loop.shutdown_asyncgens()), \
                loop.create_task(loop.shutdown_default_executor())
            for task in shutdown_tasks:
                await task
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            pending_tasks = asyncio.tasks.all_tasks(self._loop)
            for task in pending_tasks:
                task.cancel()
            try:
                self._loop.run_until_complete(_shutdown(self._loop))
            finally:
                if not self._loop.is_closed():
                    self._loop.close()

    async def _load_one(self, index: int):
        name = f'{index}.png'
        img = await self._loop.run_in_executor(
            None,
            cv2.imread,
            str(self._img_dir / name),
            cv2.IMREAD_GRAYSCALE
        )
        label = await self._loop.run_in_executor(
            None,
            cv2.imread,
            str(self._label_dir / name),
            cv2.IMREAD_GRAYSCALE
        )
        if self._pre_process is not None:
            img, label = self._pre_process((img, label))
        if callable(self._refresh):
            self._refresh()
        return (
            Tensor(img).contiguous(),
            Tensor(label).contiguous(),
        )

    def load_one(self, index: int = None):
        index = index or random.randint(1, self._length)
        if index > self._length:
            raise IndexError('index out of range')
        name = f'{index}.png'
        img = cv2.imread(str(self._img_dir / name), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(self._label_dir / name), cv2.IMREAD_GRAYSCALE)
        if self._pre_process is not None:
            img, label = self._pre_process((img, label))
        return (
            Tensor(np.expand_dims(img, axis=0)),
            Tensor(np.expand_dims(label, axis=0)),
        )

    def _load_data(self):
        def callback(fut):
            if not fut.cancelled() and index < self._length:
                load_batch()

        def load_batch():
            nonlocal index
            if index + batch_size > self._length:
                size = self._length - index + 1
            else:
                size = batch_size
            for i in range(size):
                fut = asyncio.run_coroutine_threadsafe(
                    self._load_one(i + index),
                    self._loop
                )
                self._data.add(fut)
            index += size
            fut.add_done_callback(callback)
        batch_size = self._batch_size * 2
        index = 1
        load_batch()

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        if index >= self._length:
            raise IndexError('index out of range')
        while not self._data:
            pass
        data = self._data.pop().result(timeout=None)
        return data

    def __len__(self):
        return self._length
