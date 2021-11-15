import dataclasses
from collections import namedtuple
from typing import Any, Callable, Iterable, NamedTuple, Optional, TypeVar

import numpy as np
from scipy.integrate import OdeSolution
from scipy.integrate._ivp.ivp import OdeResult as _OdeResult

diffeq = TypeVar("diffeq", bound=Callable[[float, np.ndarray, Any], np.ndarray])

# create a named tuple for hinting of result object from ODE solver
# see `scipy.stats.solve_ivp` for return object definition
SolverResultNT = namedtuple(
    "SolverResult",
    "t y sol t_events y_events nfev njev nlu status message success",
)


class OdeResult(_OdeResult):
    def __add__(self, other: SolverResultNT):
        return OdeResult(
            t=np.append(self.t, other.t),
            y=np.append(self.y, other.y, axis=1),
            sol=other.sol,
            t_events=np.append(self.t_events, other.t_events),
            y_events=np.append(self.y_events, other.y_events),
            nfev=self.nfev + other.nfev,
            njev=self.njev + other.njev,
            nlu=self.nlu + other.nlu,
            status=other.status,
            message=self.message + "\t" + other.message,
            success=self.success and other.success,
        )


@dataclasses.dataclass
class SolverResult(OdeResult):
    """Store results from the solver.

    Allows adding of results.
    """

    t: np.ndarray
    y: np.ndarray
    sol: Optional[OdeSolution]
    t_events: Optional[Iterable[np.ndarray]]
    y_events: Optional[Iterable[np.ndarray]]
    nfev: int
    njev: int
    nlu: int
    status: int
    message: str
    success: bool

    def __add__(self, other: SolverResultNT):
        return OdeResult.__add__(self, other)
