""""""

import numpy as np
import logging

from collections import namedtuple
from typing import Callable, Tuple, Union

from numpy.random import default_rng
from scipy.stats import powerlaw

from opdynamics.utils.distributions import negpowerlaw
from opdynamics.integrate.types import diffeq

logger = logging.getLogger("echo chamber")

# create a named tuple for hinting of result object from ODE solver
# see `scipy.stats.solve_ivp` for return object definition
EchoChamberSimResult = namedtuple(
    "EchoChamberSimResult",
    "t y sol t_events y_events nfev njev nlu status message success",
)


class EchoChamber(object):
    """
    A network of agents interacting with each other.

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
    def __init__(self, N, m, K, alpha, name="echochamber", seed=1337, *args, **kwargs):
        from opdynamics.socialinteraction import SocialInteraction

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
        assert alpha >= 0
        assert K >= 0

        # create array variables
        self.opinions: np.ndarray = None
        self.adj_mat: SocialInteraction = None
        self.activities: np.ndarray = None
        self.p_conn: np.ndarray = None
        self.dy_dt: diffeq = None
        self.result: EchoChamberSimResult = None
        self.init_opinions()

    def init_opinions(self, min_val=-1.0, max_val=1.0):
        """Randomly initialise opinions for all N agents between [min_val, max_val] from a uniform distribution

        :param min_val: lowest value (inclusive)
        :param max_val: highest value (inclusive)
        """
        self.opinions = self.rn.uniform(min_val, max_val, size=self.N)

    def set_activities(self, distribution=negpowerlaw, *dist_args, dim: int = 1):
        """Sample activities from a given distribution

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
            logger.warning("2D activities not tested!")
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

        .. math::
            p_{ij} = \\frac{|x_i - x_j|^{-\\beta}}{\sum_j |x_i - x_j|^{-\\beta}}

        :param beta: Power law decay of connection probability. Decay when beta>0, increase when beta<0.
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
        """Define the social interactions that occur at each time step.

        Populates `self.adj_mat` (adjacency matrix) that is used in opinion dynamics.

        :param r: Probability of a mutual interaction [0,1].
        :param lazy: Generate self.adj_mat on-demand during simulation (True) or computer all the interaction matrices
            for all time steps before (False).
        :param dt: Time step being used in simulation. Specified here so interaction dynamics have a clear time step
            even if the integration of the opinion dynamics occurs at smaller time steps (e.g. with the RK method).
        :param t_end: Last time point. Together with dt, determines the size of the social interaction array.

        """
        from opdynamics.socialinteraction import SocialInteraction

        if self.activities is None or self.p_conn is None:
            raise RuntimeError(
                """Activities and connection probabilities need to be set. 
                                                                        ec = EchoChamber(...)
                                                                        ec.set_activities(...)
                                                                        ec.set_connection_probabilities(...)
                                                                        ec.set_social_interactions(...)
                                                                        """
            )

        self.adj_mat = SocialInteraction(self, r)
        if not lazy:
            if t_end is None or dt is None:
                raise RuntimeError(
                    "`t_end` and `dt` need to be defined for eager calculation of adjacency matrix in "
                    "`set_social_interactions`"
                )
            self.adj_mat.eager(t_end, dt)

    def set_dynamics(self):
        """Set the dynamics of network by assigning a function to `self.dy_dt`.

        `self.dy_dt` is a function to be called by the ODE solver, which expects a signature of (t, y, *args).
        """

        def dy_dt(t: float, y: np.ndarray, *args) -> np.ndarray:
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

    def run_network(
        self, dt: float = 0.01, t_end: float = 0.05, method: str = "RK45"
    ) -> None:
        """Run a simulation for the echo chamber until `t_end` with a time step of `dt`.

        Because the echo chamber has ODE dynamics, an appropriate method should be chosen from
        `scipy.integrate.solver_ivp` or `opdynamics.integrate.solvers`

        :param dt: (Max) Time step for integrator. Smaller values will yield more accurate results but the simulation
            will take longer. Large `dt` for unstable methods (like "Euler") can cause numerical instability where
            results show **increasingly large** oscillations in opinions (nonsensical).
        :param t_end: Time for simulation to span. Number of iterations will be at least t_end/dt.
        :param method: Integration method to use. Must be one specified by `scipy.integrate.solver_ivp` or
            `opdynamics.integrate.solvers`

        """
        from scipy.integrate import solve_ivp
        from opdynamics.integrate.solvers import ODE_INTEGRATORS, solve_ode

        if (
            self.activities is None
            or self.p_conn is None
            or self.adj_mat is None
            or self.dy_dt is None
        ):
            raise RuntimeError(
                """Activities, connection probabilities, social interactions, and dynamics need to be set. 
                                                                        ec = EchoChamber(...)
                                                                        ec.set_activities(...)
                                                                        ec.set_connection_probabilities(...)
                                                                        ec.set_social_interactions(...)
                                                                        ec.set_dynamics(...)
                                                                        ec.run_network(...)
                                                                        """
            )

        args = (self.K, self.alpha, self.N, self.m, self.p_conn, self.adj_mat, dt)

        if method in ODE_INTEGRATORS:
            # use a custom method in `opdynamics.utils.integrators`
            self.result: EchoChamberSimResult = solve_ode(
                self.dy_dt,
                t_span=[0, t_end],
                y0=self.opinions,
                method=method,
                dt=dt,
                args=args,
            )
        else:
            # use a method in `scipy.integrate`
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

        `t` can be an array of numbers if it is a numpy ndarray.

        If `t` is -1, the last time point is used.

        If `t` is None, all time points are retrieved


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

    def get_nearest_neighbours(self):
        """Calculate mean value of every agents' nearest neighbour.

        .. math::
                \\frac{\sum_j a_{ij} x_j}{\sum_j a_{ij}}

        where

        .. math:: a_{ij}
        represents the (static) adjacency matrix of the aggregated interaction network
        and

        .. math:: \sum_j a_{ij}
        is the degree of node `i`.

        """
        snapshot_adj_mat = self.adj_mat[-1]
        out_degree_i = np.sum(snapshot_adj_mat, axis=0)
        close_opinions = np.sum(snapshot_adj_mat * self.opinions, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            # suppress warnings about dividing by nan or 0
            nn = close_opinions / out_degree_i
        return nn

    # noinspection PySameParameterValue
    @staticmethod
    def run_params(
        N: int = 1000,
        m: int = 10,
        K: float = 3,
        alpha: float = 2,
        beta: float = 2,
        epsilon: float = 1e-2,
        gamma: float = 2.1,
        dt: float = 0.01,
        T: float = 1.0,
        r: float = 0.5,
        method: str = "RK45",
        plot_opinion: Union[bool, str] = False,
        lazy: bool = False,
    ):
        """Class method to quickly and conveniently run a simulation where the parameters differ, but the structure
        is the same (activity distribution, dynamics, etc.)"""
        logger.debug(
            f"run_params(N={N}, m={m}, K={K}, alpha={alpha}, beta={beta}, epsilon={epsilon}, gamma={gamma}, "
            f"dt={dt}, T={T}, r={r}, plot_opinion={plot_opinion}, lazy={lazy})"
        )
        _ec = EchoChamber(N, m, K, alpha)
        _ec.set_activities(negpowerlaw, gamma, epsilon, 1, dim=1)
        _ec.set_connection_probabilities(beta=beta)
        _ec.set_social_interactions(r=r, lazy=lazy, dt=dt, t_end=T)
        _ec.set_dynamics()
        _ec.run_network(dt=dt, t_end=T, method=method)
        if plot_opinion:
            from matplotlib.axes import Axes
            from opdynamics.visualise import VisEchoChamber

            _vis = VisEchoChamber(_ec)
            if plot_opinion == "summary":
                _vis.show_summary()
            else:
                _vis.show_opinions(True)
        return _ec


class NoisyEchoChamber(EchoChamber):
    # noinspection PyTypeChecker
    def __init__(self, name="noisy echochamber", *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.diffusion: diffeq = None
        self.wiener_process: Callable = None

    def set_dynamics(self, D=0.01):
        """Set up Stochastic ordinary differential equation. See `run_network` for changes to the integration."""
        # assign drift as before, aka dy_dt
        super().set_dynamics()

        # create new diffusion term
        self.diffusion = lambda t, y, *args: np.sqrt(D)
        self.wiener_process = lambda: self.rn.normal(0, 1, size=self.N)

    def run_network(self, dt=0.01, t_end=0.05, method="Euler–Maruyama", r=None):
        """Dynamics are no longer of an ordinary differential equation so we can't use scipy.solve_ivp anymore"""

        from opdynamics.integrate.solvers import SDE_INTEGRATORS, solve_sde

        if (
            self.activities is None
            or self.p_conn is None
            or self.adj_mat is None
            or self.dy_dt is None
        ):
            raise RuntimeError(
                """Activities, connection probabilities, social interactions, and dynamics need to be set. 
                                                                            ec = EchoChamber(...)
                                                                            ec.set_activities(...)
                                                                            ec.set_connection_probabilities(...)
                                                                            ec.set_social_interactions(...)
                                                                            ec.set_dynamics(...)
                                                                            ec.run_network(...)
                                                                            """
            )

        args = (self.K, self.alpha, self.N, self.m, self.p_conn, self.adj_mat, dt)

        if method in SDE_INTEGRATORS:
            # use a custom method in `opdynamics.utils.integrators`
            self.result: EchoChamberSimResult = solve_sde(
                self.dy_dt,
                self.diffusion,
                self.wiener_process,
                t_span=[0, t_end],
                y0=self.opinions,
                method=method,
                dt=dt,
                args=args,
            )
        else:
            raise NotImplementedError()
        # reassign opinions to last time point
        self.opinions = self.result.y[:, -1]
        logger.debug(f"done running {self.name}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from opdynamics.visualise import VisEchoChamber

    logging.basicConfig(level=logging.DEBUG)

    num_agents = 1000
    m = 10  # number of other agents to interact with
    alpha = 2  # controversialness of issue (sigmoidal shape)
    K = 3  # social interaction strength
    epsilon = 1e-2  # minimum activity level with another agent
    gamma = 2.1  # power law distribution param
    beta = 2  # power law decay of connection probability
    activity_distribution = negpowerlaw

    dt = 0.01
    t_end = 0.5

    ec = EchoChamber(num_agents, m, K, alpha, seed=1337)
    vis = VisEchoChamber(ec)

    ec.set_activities(activity_distribution, gamma, epsilon, 1)
    vis.show_activities()

    ec.set_connection_probabilities(beta=beta)
    ec.set_social_interactions(r=0.5, dt=dt, t_end=t_end)
    ec.set_dynamics()

    ec.run_network(dt=dt, t_end=t_end)
    vis.show_opinions(color_code=False)

    # this is shorthand for above
    EchoChamber.run_params(
        num_agents, m, K, alpha, beta, epsilon, gamma, 0.01, 0.5, plot_opinion=True
    )

    EchoChamber.run_params(
        num_agents, m, K, alpha, beta, epsilon, gamma, 0.01, 1, plot_opinion="summary"
    )

    plt.show()
