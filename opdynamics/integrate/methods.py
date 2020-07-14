import logging

from abc import ABCMeta, abstractmethod
from typing import Callable

import numpy as np

from opdynamics.integrate.types import diffeq

logger = logging.getLogger("integration methods")


class ODEIntegrator(metaclass=ABCMeta):
    """Abstract class for custom Ordinary Differential Equation (ODE) integration methods.

    * dy_dt - function body is expected to be in the form `y_dot = -y + ...` and have signature `(t, y, *args)`.
    * y0 - starting value of y (a copy if made so the original reference is not mutated).
    * args - additional arguments to be passed to `dy_dt`.
    """

    def __init__(self, dy_dt: diffeq, y0: np.ndarray, args: tuple):
        self.dy_dt = dy_dt
        self.y = y0.copy()
        self.args = args

    @abstractmethod
    def step(self, t: float, dt: float):
        """Calculate new y from a single step of an ODE in the form

        `dy_dt = -y + ...`

        """
        raise NotImplementedError("must implement `step_size`.")


class Euler(ODEIntegrator):
    """Forward Euler integration"""

    def step(self, t: float, dt: float):
        """Calculate new y from a single step of an ODE in the form

        :math:`\\frac{dy}{dt} = -y + ...`

        """
        self.y += self.dy_dt(t, self.y, *self.args) * dt


class SDEIntegrator(metaclass=ABCMeta):
    """Abstract class for custom Stochastic Differential Equation (SDE) integration methods.

    Solving one-dimensional SDEs `dy = f(t, y)dt + g(t, y)dW_t` is like an ODE except with an extra function for
    the diffusion (randomness or noise) term.

    * dy_dt - function body is expected to be in the form `y_dot = -y + ...` and have signature `(t, y, *args)`.
    * diffusion - function is amount of noise in the system.
    * wiener_process - function for generating noise. Should be defined in terms of EchoChamber's properties:
        e.g. `self.rn.normal(0, 1, size=self.N)`
    * y0 - starting value of y (a copy if made so the original reference is not mutated).
    * args - additional arguments to be passed to `dy_dt` and `diffusion`.

    """

    def __init__(
        self,
        dy_dt: diffeq,
        diffusion: diffeq,
        wiener_process: Callable,
        y0: np.ndarray,
        args: tuple,
        diff_args: tuple,
    ):
        self.dy_dt = dy_dt
        self.diffusion = diffusion
        self.wiener_process = wiener_process
        self.y = y0.copy()
        self.args = args
        self.diff_args = diff_args
        self.t = 0

    @abstractmethod
    def step(self, t: float, dt: float):
        raise NotImplementedError("must implement `step_size`.")


class EulerMaruyama(SDEIntegrator):
    def step(self, t: float, dt: float):
        drift = self.dy_dt(t, self.y, *self.args)
        diff = self.diffusion(t, self.y, *self.diff_args)
        self.y = self.y + drift * dt + diff * self.wiener_process(dt)
