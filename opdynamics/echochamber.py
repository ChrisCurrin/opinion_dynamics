""""""

import numpy as np
import logging

from collections import namedtuple
from typing import Callable, Tuple, Union

from numpy.random import default_rng
from scipy.stats import powerlaw

from opdynamics.utils.distributions import negpowerlaw

logging.basicConfig()
logger = logging.getLogger("echo chamber")

# create a named tuple for hinting of result object from ODE solver
# see `scipy.stats.solve_ivp` for return object definition
EchoChamberSimResult = namedtuple(
    "EchoChamberSimResult",
    "t y sol t_events y_events nfev njev nlu status message success",
)


class EchoChamber(object):
    """ A network of agents interacting with each other.

    * N - number of agents
    * m - number of other agents to interact with
    * alpha - controversialness of issue (sigmoidal shape)
    * K - social interaction strength
    * epsilon - minimum activity level with another agent
    * gamma - power law distribution param
    * beta - power law decay of connection probability
    * p_mutual_interaction - probability of a p_mutual_interaction interaction

    """

    # noinspection PyTypeChecker
    def __init__(
        self, N, m=10, K=3.0, alpha=3.0, name="echochamber", seed=1337, *args, **kwargs
    ):
        # create a random number generator for this object (to be thread-safe)
        self.rn = default_rng(seed)

        # create a human-readable name for ths object
        self.name = name

        # assign args to object variables
        self.N = N
        self.m = m
        self.K = K
        self.alpha = alpha
        # quick checks
        assert N > 0 and type(N) is int
        assert 0 < m < N and type(m) is int
        assert alpha > 0
        assert K > 0

        # create array variables
        self.opinions: np.ndarray = None
        self.adj_mat: np.ndarray = None
        self.activities: np.ndarray = None
        self.p_conn: np.ndarray = None
        self.dy_dt: Callable = None
        self.result: EchoChamberSimResult = None
        self.init_opinions()

    def init_opinions(self, min_val=-1.0, max_val=1.0):
        """Randomly initialise opinions for all N agents between [min_val, max_val] from a uniform distribution

        :param min_val: lowest value (inclusive)
        :param max_val: highest value (inclusive)
        """
        self.opinions = self.rn.uniform(min_val, max_val, size=self.N)

    def set_activities(self, distribution=negpowerlaw, *dist_args, dim: int = 1):
        """
        Sample activities from a given distribution

        :param distribution: A distribution that extends `rv_continuous`, such as `powerlaw`,
            or is like `rv_continuous` (has a `rvs` method) to retrieve random samples.

        :param dist_args: Arguments to pass to the distribution. See `scipy.stats`.
            For `powerlaw` (default), the expected arguments are (gamma, min_val, max_val)

        :param dim: Number of dimensions for the activities. First dimension is the number of agents (N),
            second dimension is the number of agents that an agent interacts with (m), further dimensions are not
            supported and raises an error.

        """
        if dim == 1:
            size = self.N
        elif dim == 2:
            size = (self.N, self.m)
        else:
            raise NotImplementedError("dimensions of more than 2 not implemented")

        if distribution == powerlaw:
            # some tinkering of arguments for `rvs` method so that we can keep this method's arguments clear
            gamma, min_val, max_val = dist_args
            # to compensate for shift
            dist_args = (*dist_args[:2], max_val - min_val)

        self.activities = distribution.rvs(*dist_args, size=size)

    def set_connection_probabilities(self, beta: float = 0.0):
        """For agent `i`, the probability of connecting to agent `j` is a function of the absolute strength of
        their opinions and a beta param, relative to all of the differences between an agent i and every other agent.

        $$ p_{ij} = \frac{|x_i - x_j|^{-\beta}}{\sum_j |x_i - x_j|^{-\beta}} $$

        :param beta: Power law decay of connection probability. Decay when beta>0, increase when beta>0.
            When beta=0, then connection probabilities are uniform.

        """
        p_conn = np.zeros(shape=(self.N, self.N))
        for i in range(self.N):
            mag = np.abs(self.opinions[i] - self.opinions)
            mag[i] = np.nan
            p_conn[i] = np.power(mag, -beta)
            p_conn[i, i] = 0
            p_conn[i] /= np.sum(p_conn[i])
        self.p_conn = p_conn

    def set_social_interactions(
        self, r: float = 0.5, lazy=False, dt: float = None, t_end: float = None
    ):
        """
        Define the social interactions that occur at each time step.

        Populates `self.adj_mat` (adjacency matrix) that is used in opinion dynamics.

        :param r: Probability of a mutual interaction [0,1].
        :param lazy: Generate self.adj_mat on-demand during simulation (True) or computer all the interaction matrices
            for all time steps before (False).
        :param dt: Time step being used in simulation. Specified here so interaction dynamics have a clear time step
            even if the integration of the opinion dynamics occurs at smaller time steps (e.g. with the RK method).
        :param t_end: Last time point. Together with dt, determines the size of the social interaction array.

        """
        from opdynamics.socialinteraction import (
            SocialInteraction,
            get_social_interaction,
        )

        if lazy:
            self.adj_mat = SocialInteraction(self, r)
        else:
            t_arr = np.arange(0, t_end + dt, dt)
            self.adj_mat = np.zeros((len(t_arr), self.N, self.N), dtype=int)
            active_thresholds = self.rn.random(size=len(t_arr))
            for t_idx, t_point in enumerate(t_arr):
                self.adj_mat[t_idx, :, :] = get_social_interaction(
                    self, active_thresholds[t_idx], r
                )

    def set_dynamics(self):
        """Set the dynamics of network by assigning a function to `self.dy_dt`.

        `self.dy_dt` is a function to be called by the ODE solver, which expects a signature of (t, y, *args).
        """

        def dy_dt(t, y, *args):
            """Activity-Driven (AD) network dynamics.

            1. get the interactions (A) that happen at this time point between each of N agents based on activity
            probabilities (p_conn) and the number of agents to interact with (m).
            2. calculate opinion derivative by getting the scaled (by social influence, alpha) opinions (y.T) of agents
            interacting with each other (A), multiplied by social interaction strength (K).

            """
            K, alpha, N, m, p_conn, A, dt = args
            # get activity matrix for this time point (pre-generated)
            At = A[int(t / dt)]
            return -y.T + K * np.sum(At * np.tanh(alpha * y.T), axis=1)

        self.dy_dt = dy_dt

    def run_network(self, dt=0.01, t_end=0.05, method="RK45"):
        from scipy.integrate import solve_ivp

        if self.activities is None or self.p_conn is None or self.dy_dt is None:
            raise RuntimeError(
                """Activities, connection probabilities, and dynamics need to be set. 
                                                            ec = EchoChamber(...)
                                                            ec.set_activities(...)
                                                            ec.set_connection_probabilities(...)
                                                            ec.set_dynamics(...)
                                                            ec.run_network(...)
                                                            """
            )

        if self.adj_mat is None:
            self.set_social_interactions(r=True, dt=dt, t_end=t_end)

        args = (self.K, self.alpha, self.N, self.m, self.p_conn, self.adj_mat, dt)

        if method == "FEuler":
            # forward euler
            y = self.opinions
            t_arr = np.arange(0, t_end + dt, dt)
            y_arr = np.zeros(shape=(len(t_arr), len(y)))
            for i, t in enumerate(t_arr):
                y_arr[i] = y
                dy_dt = self.dy_dt(t, y, *args)
                dy = dy_dt * dt
                y += dy
            # add final y
            y_arr[-1] = y
            self.result = EchoChamberSimResult(
                t_arr, y_arr.T, None, None, None, None, None, None, None, None, True
            )
        else:
            self.result: EchoChamberSimResult = solve_ivp(
                self.dy_dt,
                t_span=[0, t_end],
                y0=self.opinions,
                method=method,
                vectorized=True,
                args=args,
                first_step=dt,
                max_step=dt,
            )
        # reassign opinions to last time point
        self.opinions = self.result.y[:, -1]
        logger.debug(f"done running {self.name}")

    def get_mean_opinion(
        self, t: Union[np.number, np.ndarray] = -1
    ) -> Tuple[float, np.ndarray]:
        """Calculate the average opinion at time point `t`.

        t can be an array of numbers if it is a numpy ndarray.

        If t is -1, the last time point is used.

        If t is None, all time points are retrieved


        :param t: time point to get the average for. The closest time point is used.
        :return: pair of actual time point(s) used, mean value(s) of opinions at actual time point(s).
        """
        if self.result is None:
            raise RuntimeError(
                f"{self.name} has not been run. call `.run_network` first."
            )
        if isinstance(t, np.number):
            idx = -1 if t == -1 else np.argmin(np.abs(t - self.result.t))
        else:
            idx = t if t is not None else np.arange(len(self.result.t))
        time_point, average = self.result.t[idx], np.mean(self.result.y[:, idx], axis=0)
        return time_point, average

    # noinspection PySameParameterValue
    @staticmethod
    def run_params(
        N=1000,
        m=10,
        K=3,
        alpha=0.05,
        beta=2,
        epsilon=1e-2,
        gamma=2.1,
        dt=0.01,
        T=1.0,
        mutual_interactions=0.5,
        plot_opinion=False,
    ):
        """Class method to quickly and conveniently run a simulation where the parameters differ, but the structure
        is the same (activity distribution, dynamics, etc.)"""
        logger.debug(
            f"run_params(N={N}, m={m}, K={K}, alpha={alpha}, beta={beta}, epsilon={epsilon}, gamma={gamma}, "
            f"dt={dt}, T={T}, mutual_interactions={mutual_interactions}, plot_opinion={plot_opinion})"
        )
        _ec = EchoChamber(N, m, K, alpha)
        _ec.set_activities(negpowerlaw, gamma, epsilon, 1, dim=1)
        _ec.set_connection_probabilities(beta=beta)
        _ec.set_social_interactions(r=mutual_interactions, dt=dt, t_end=T)
        _ec.set_dynamics()
        _ec.run_network(dt=dt, t_end=T, method="RK45")
        if plot_opinion:
            from matplotlib.axes import Axes
            from opdynamics.visualise import VisEchoChamber

            _vis = VisEchoChamber(_ec)
            _vis.show_opinions(
                True, ax=plot_opinion if isinstance(plot_opinion, Axes) else None
            )
        return _ec


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from opdynamics.visualise import VisEchoChamber

    num_agents = 1000
    m = 10  # number of other agents to interact with
    alpha = 0.05  # controversialness of issue (sigmoidal shape)
    K = 3  # social interaction strength
    epsilon = 1e-2  # minimum activity level with another agent
    gamma = 2.1  # power law distribution param
    beta = 2  # power law decay of connection probability
    activity_distribution = negpowerlaw
    ec = EchoChamber(num_agents, m, K, alpha, seed=1337)
    vis = VisEchoChamber(ec)

    ec.set_activities(activity_distribution, gamma, epsilon, 1)
    vis.show_activities()

    ec.set_connection_probabilities(beta=beta)
    ec.set_dynamics()

    ec.run_network(dt=0.01, t_end=0.5)
    vis.show_opinions(color_code=False)

    # this is shorthand for above
    EchoChamber.run_params(
        num_agents, m, K, alpha, beta, epsilon, gamma, 0.01, 0.5, True, True
    )

    plt.show()
