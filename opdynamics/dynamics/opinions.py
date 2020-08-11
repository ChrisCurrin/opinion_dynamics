"""
Methods that define how opinions evolve over time.

- must be hashable (i.e. all functions are top-level) for asynchronous programming
- must have a call signature of ``t, y, *args`` for compatibility with scipy's ODE solvers
    ``*args`` is specified by ``EchoChamber._args``
"""
import numpy as np

from opdynamics.metrics.opinions import sample_means


def dy_dt(t: float, y: np.ndarray, *args) -> np.ndarray:
    """Opinion dynamics.

    1. get the interactions (A) that happen at this time point between each of N agents based on activity
    probabilities (p_conn) and the number of agents to interact with (m).

    2. calculate opinion derivative by getting the scaled (by social influence, alpha) opinions (y.T) of agents
    interacting with each other (A), multiplied by social interaction strength (K).

    """
    K, alpha, A, dt = args
    # get activity matrix for this time point
    At = A[int(t / dt)]
    return -y.T + K * np.sum(At * np.tanh(alpha * y.T), axis=1)


def _new_clt_sample(ec, y: np.ndarray, n: int, num_samples: int):
    """
    method to re-assign self._sample_means
    :math:`\\sqrt {n}\\left({\\bar{X}}_{n}-\\mu \\right) \\rightarrow \mathcal{N}\\left(0,\\sigma ^{2}\\right)`

    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    :param ec: EchoChamber object to store sample means (and to use it's random number generator)
    :type ec: opdynamics.networks.EchoChamber
    :param y: opinions
    :param n: sample size
    :param num_samples: number of samples
    """
    ec._sample_means = np.sqrt(n) * (
        sample_means(y, n, num_samples=num_samples, rng=ec.rn) - np.mean(y)
    )


def sample_dy_dt(t: float, y: np.ndarray, *all_args) -> np.ndarray:
    """Opinion dynamics with random opinion samples.

    1 - 3 from either `dy_dt` or `dynamic_conn` (specified by `ec.super_dy_dt`).

    4. add a "population opinion" term that captures the Lindeberg–Lévy Central Limit Theorem -
    :math:`\\sqrt {n}\\left({\\bar{X}}_{n}-\\mu \\right) \\rightarrow \mathcal{N}\\left(0,\\sigma ^{2}\\right)`
    \\
    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    """
    ec, n, num_samples, *other_args = all_args
    if type(n) is tuple:
        # choose between low and high values (randint not implemented for default_rng)
        n = ec.rn.choice(np.arange(n[0], n[1], dtype=int))
    if np.round(t % other_args[-1], 6) == 0:
        # calculate sample means every explicit dt (other_args[-1]) - independent of solver's dt
        _new_clt_sample(ec, y, n, num_samples)
    return ec.super_dy_dt(t, y, *other_args) + ec.D * ec._sample_means