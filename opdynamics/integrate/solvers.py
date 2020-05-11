import logging
import numpy as np
from typing import Callable
from tqdm import tqdm
from opdynamics.echochamber import EchoChamberSimResult
from opdynamics.integrate.methods import Euler, EulerMaruyama

logger = logging.getLogger("solvers")

ODE_INTEGRATORS = {"Euler": Euler}
SDE_INTEGRATORS = {"Euler-Maruyama": EulerMaruyama, "EM": EulerMaruyama}


def _run_solver(solver, t_span: list, dt: float) -> EchoChamberSimResult:
    """Given a numerical integrator, call its 'step' method T/dt times (where T is last element of t_span)."""
    t_end = t_span[-1]
    t_arr = np.arange(0, t_end + dt, dt)
    y_arr = np.zeros(shape=(len(t_arr), len(solver.y)))
    for i, t in tqdm(enumerate(t_arr)):
        y_arr[i] = solver.y
        solver.step(t, dt)
    # add final y
    y_arr[-1] = solver.y
    return EchoChamberSimResult(
        t_arr, y_arr.T, None, None, None, None, None, None, None, None, True
    )


def solve_ode(
    dy_dt: Callable, t_span: list, y0: np.ndarray, method: str, dt: float, args: tuple
) -> EchoChamberSimResult:
    """Solve an ordinary differential equation using a method in `opdynamics.integrate.methods`."""
    if method in ODE_INTEGRATORS:
        logger.debug(f"solving ODE using {method}")
        solver = ODE_INTEGRATORS[method](dy_dt, y0, args)
        return _run_solver(solver, t_span, dt)
    else:
        raise NotImplementedError(
            f"ODE integration method '{method}' not implemented. Supported methods: {ODE_INTEGRATORS}"
        )


def solve_sde(
    dy_dt: Callable,
    diffusion: Callable,
    wiener_process: Callable,
    t_span: list,
    y0: np.ndarray,
    method: str,
    dt: float,
    args: tuple,
) -> EchoChamberSimResult:
    """Solve a stochastic ordinary differential equation using a method in `opdynamics.integrate.methods`."""
    if method in SDE_INTEGRATORS:
        logger.debug(f"solving SDE using {method}")
        solver = SDE_INTEGRATORS[method](dy_dt, diffusion, wiener_process, y0, args)
        return _run_solver(solver, t_span, dt)
    else:
        raise NotImplementedError(
            f"SDE integration method '{method}' not implemented. Supported methods: {ODE_INTEGRATORS}"
        )
