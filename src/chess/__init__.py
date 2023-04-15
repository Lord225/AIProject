from typing import Tuple
import numpy as np
import numba as nb

import chess


class DiagonalChess:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action) -> Tuple[np.ndarray, int, bool]:
        """
        ## returns
        - observation: np.ndarray
        - reward: int
        - done: bool
        """

        raise NotImplementedError

    def render(self):
        pass








