"""
Methods that define how opinions evolve over time.

- must be hashable (i.e. all functions are top-level) for asynchronous programming
- must have a call signature of ``t, y, *args`` for compatibility with scipy's ODE solvers
    ``*args`` is specified by ``EchoChamber._args``
"""
import numpy as np

from opdynamics.metrics.opinions import sample_means

##############################
# Opinion dynamics
##############################

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

def sample_dy_dt(t: float, y: np.ndarray, *all_args) -> np.ndarray:
    """Opinion dynamics with random dynamical nudge.

    1 - 3 from either `dy_dt` or `dynamic_conn` (specified by `ec.super_dy_dt`).

    4. add a "population opinion" term that captures the Lindeberg–Lévy Central Limit Theorem -
    :math:`\\sqrt {n}\\left({\\bar{X}}_{n}-\\mu \\right) \\rightarrow \mathcal{N}\\left(0,\\sigma ^{2}\\right)`
    \\
    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    """
    clt_sample_method, ec, n, num_samples, *other_args = all_args
    if np.round(t % other_args[-1], 6) == 0:
        # calculate sample means every explicit dt (other_args[-1]) - independent of solver's dt
        if type(n) is tuple:
            # choose between low and high values (randint not implemented for default_rng)
            n = ec.rn.choice(np.arange(n[0], n[1], dtype=int))
        clt_sample_method(ec, y, n, num_samples)
    return ec.super_dy_dt(t, y, *other_args) + ec.D * ec._sample_means


def sample_dy_dt_activity(t: float, y: np.ndarray, *all_args) -> np.ndarray:
    """Opinion dynamics with random opinion samples.

    As with :meth:`sample_dy_dt`, but the RDN is gated by the adjency matrix (determined by agent activity)

    """
    clt_sample_method, ec, n, num_samples, *other_args = all_args
    K, alpha, A, dt = other_args
    if np.round(t % dt, 6) == 0:
        # calculate sample means every explicit dt (other_args[-1]) - independent of solver's dt
        if type(n) is tuple:
            # choose between low and high values (randint not implemented for default_rng)
            n = ec.rn.choice(np.arange(n[0], n[1], dtype=int))
        clt_sample_method(ec, y, n, num_samples)

    return ec.super_dy_dt(t, y, *other_args) + ec.D * np.sum(
        A[int(t / dt)] * ec._sample_means, axis=1
    )


##############################
# Sampling approaches (used in SampleChamber)
#   Note that :meth:`ec._sample_means` is used to store the result for subsequent retrieval at non `dt` time points.
#       A more elegant solution may be possible in the future, akin perhaps to the adjency matrix implementation.
##############################

def _basic_clt_sample(ec, y: np.ndarray, n: int, num_samples: int):
    """
    method to re-assign self._sample_means
    :math:`\\sqrt {n}\\left(\\bar{X}_{n}-\\mu \\right) \\rightarrow \\mathcal{N}\\left(0,\\sigma ^{2}\\right)`

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


def _outer_sigmoid_clt_sample(ec, y: np.ndarray, n: int, num_samples: int):
    """
    method to re-assign self._sample_means
    :math:`\\tanh{(\\sqrt{n}\\left(\\bar{X}_{n}-\\mu \\right)})`

    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    :param ec: EchoChamber object to store sample means (and to use it's random number generator)
    :type ec: opdynamics.networks.EchoChamber
    :param y: opinions
    :param n: sample size
    :param num_samples: number of samples
    """
    ec._sample_means = np.tanh(
        np.sqrt(n)
        * (sample_means(y, n, num_samples=num_samples, rng=ec.rn) - np.mean(y))
    )


def _inner_sigmoid_clt_sample(ec, y: np.ndarray, n: int, num_samples: int):
    """
    method to re-assign self._sample_means
    :math:`\\tanh{(\\sqrt{n}\\left(\\bar{X}_{n}-\\mu \\right)})`

    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    :param ec: EchoChamber object to store sample means (and to use it's random number generator)
    :type ec: opdynamics.networks.EchoChamber
    :param y: opinions
    :param n: sample size
    :param num_samples: number of samples
    """
    ec._sample_means = np.sqrt(n) * np.tanh(
        sample_means(y, n, num_samples=num_samples, rng=ec.rn) - np.mean(y)
    )


def _subsample_clt_sample(ec, y: np.ndarray, n: int, num_samples: int):
    """
    method to re-assign self._sample_means
    :math:`D \\cdot \\tanh(x_k - \\bar{X_n})`

    where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

    :param ec: EchoChamber object to store sample means (and to use it's random number generator)
    :type ec: opdynamics.networks.EchoChamber
    :param y: opinions
    :param n: sample size
    :param num_samples: number of samples
    """
    ec._sample_means = sample_means(
        y, 1, num_samples=num_samples, rng=ec.rn
    ) - sample_means(y, n, num_samples=num_samples, rng=ec.rn)


clt_methods = {
    "basic": _basic_clt_sample,
    "outer_sigmoid": _outer_sigmoid_clt_sample,
    "inner_sigmoid": _inner_sigmoid_clt_sample,
    "subsample": _subsample_clt_sample,
}
clt_methods.setdefault(None, _basic_clt_sample)

