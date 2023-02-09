from functools import lru_cache
from pathlib import Path
import threading
import asyncio

from torch.utils.data import Dataset
import cv2
import numpy as np

from . import loader
from .. import config
from ..logging import logger


class Dataset(Dataset):
    def __init__(self):
        self._refresh = None
        self._data_path = data_path = Path('./data')
        config_for_data = config.config_for_data
        if not data_path.exists() or config_for_data.reinit:
            if config_for_data.reinit:
                logger.info('reinit data...')
            else:
                logger.info('not find data in path, initialize it...')
            loader.init(
                config_for_data.image_num,
                config_for_data.image_size,
                config_for_data.min_circle_num,
                config_for_data.max_circle_num,
                config_for_data.min_circle_size,
                config_for_data.max_circle_size,
                config_for_data.theta_step,
                config_for_data.start_angle,
                config_for_data.end_angle,
            )
            logger.info('data successfully initialized')
        self._data = set()
        self._img_dir = self._data_path / 'transformed_imgs'
        self._label_dir = self._data_path / 'imgs'
        self._length = config_for_data.image_num
        self._loop = asyncio.new_event_loop()
        self._batch_size = config.config_for_train.batch_size

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
        if callable(self._refresh):
            self._refresh()
        img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        label = cv2.normalize(label, None, -0.5, 0.5, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return (
            np.expand_dims(img, axis=0),
            np.expand_dims(label, axis=0),
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
