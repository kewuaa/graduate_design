from os import popen
import sys
import time

import visdom
import numpy as np

from ..logging import logger


class Visualizer:
    def __init__(self, env: str = 'default'):
        try:
            logger.info('connecting visdom server')
            self._visdom = visdom.Visdom(env=env, raise_exceptions=True)
            logger.info('connect visdom server failed')
        except Exception:
            logger.info('try to open the visdom server')
            cmd = f'start {sys.executable} -m visdom.server'
            popen(cmd)
            try:
                self._visdom = visdom.Visdom(env=env, raise_exceptions=True)
            except Exception as e:
                raise e
        self._visdom.text('start logging......', win='logging')

    def plot(self, x, y, win: str):
        self._visdom.line(
            Y=np.array([y]),
            X=np.array([x]),
            win=win,
            update='append' if x else None
        )

    def log(self, text: str, win: str = 'logging'):
        log_text = f'[{time.strftime("%m%d_%H%M%S")}] {text}'
        self._visdom.text(log_text, win=win, append=True)
