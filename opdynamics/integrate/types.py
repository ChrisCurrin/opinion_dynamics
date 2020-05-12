from collections import namedtuple
from typing import Any, Callable, TypeVar

import numpy as np

diffeq = TypeVar("diffeq", bound=Callable[[float, np.ndarray, Any], np.ndarray])

# create a named tuple for hinting of result object from ODE solver
# see `scipy.stats.solve_ivp` for return object definition
SolverResult = namedtuple(
    "SolverResult", "t y sol t_events y_events nfev njev nlu status message success",
)
