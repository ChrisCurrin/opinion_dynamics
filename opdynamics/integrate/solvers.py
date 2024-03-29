import logging
import numpy as np
from typing import Callable, Tuple
from tqdm.contrib import tenumerate

from opdynamics.integrate.types import SolverResult
from opdynamics.integrate.methods import Euler, EulerMaruyama

logger = logging.getLogger("solvers")

ODE_INTEGRATORS = {"Euler": Euler}
SDE_INTEGRATORS = {"Euler-Maruyama": EulerMaruyama}


def _run_solver(
    solver,
    t_span: Tuple[float, float],
    dt: float,
    desc: str = "solver",
    show_pbar=True,
) -> SolverResult:
    """Given a numerical integrator, call its 'step' method T/dt times (where T is last element of t_span)."""
    t_start, t_end = t_span
    t_arr = np.round(np.arange(t_start, t_end + dt, dt), 6)
    y_arr = np.zeros(shape=(len(t_arr), len(solver.y)))
    logger.debug(f"{len(t_arr)} iterations to do...")

    for i, t in tenumerate(t_arr, desc=desc, disable=(not show_pbar)):
        y_arr[i] = solver.y
        solver.step(t, dt)
    # add final y
    y_arr[-1] = solver.y
    return SolverResult(t_arr, y_arr.T, None, None, None, 0, 0, 0, 1, "success", True)


def solve_ode(
    dy_dt: Callable,
    t_span: tuple,
    y0: np.ndarray,
    method: str,
    dt: float,
    args: tuple,
    **kwargs,
) -> SolverResult:
    """Solve an ordinary differential equation using a method in `opdynamics.integrate.methods`."""
    if method in ODE_INTEGRATORS:
        logger.debug(f"solving ODE using {method}")
        solver = ODE_INTEGRATORS[method](dy_dt, y0, args)
        return _run_solver(solver, t_span, dt, **kwargs)
    else:
        raise NotImplementedError(
            f"ODE integration method '{method}' not implemented. Supported methods: {ODE_INTEGRATORS}"
        )


def solve_sde(
    dy_dt: Callable,
    diffusion: Callable,
    t_span: tuple,
    y0: np.ndarray,
    method: str,
    dt: float,
    args: tuple,
    diff_args: tuple,
    **kwargs,
) -> SolverResult:
    """Solve a stochastic ordinary differential equation using a method in `opdynamics.integrate.methods`."""
    if method in SDE_INTEGRATORS:
        logger.debug(f"solving SDE using {method}")
        solver = SDE_INTEGRATORS[method](dy_dt, diffusion, y0, args, diff_args)
        return _run_solver(solver, t_span, dt, **kwargs)
    else:
        raise NotImplementedError(
            f"SDE integration method '{method}' not implemented. Supported methods: {ODE_INTEGRATORS}"
        )
