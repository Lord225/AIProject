from collections import namedtuple
from typing import Tuple
from tensorflow import Tensor

ReplayHistoryType = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]