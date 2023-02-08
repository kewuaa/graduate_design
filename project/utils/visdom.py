import time

import visdom
import numpy as np


class Visualizer:
    def __init__(self, env: str = 'default'):
        self._visdom = visdom.Visdom(env=env)
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
