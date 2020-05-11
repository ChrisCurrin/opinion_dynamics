from typing import Any, Callable, TypeVar

import numpy as np

diffeq = TypeVar("diffeq", bound=Callable[[float, np.ndarray, Any], np.ndarray])
