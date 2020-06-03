""""""
import copy
import os

import numpy as np
import pandas as pd
import logging

from typing import Callable, Tuple, Union

from numpy.random import default_rng
from scipy.stats import powerlaw

from opdynamics.utils.accuracy import precision_and_scale
from opdynamics.utils.constants import (
    EXTERNAL_NOISE,
    INTERNAL_NOISE,
    INTERNAL_NOISE_SIG,
    INTERNAL_NOISE_SIG_K,
)
from opdynamics.utils.decorators import hashable
from opdynamics.utils.distributions import negpowerlaw
from opdynamics.integrate.types import SolverResult, diffeq
from opdynamics.utils.errors import ECSetupError

logger = logging.getLogger("echo chamber")


@hashable
class EchoChamber(object):
    """
    A network of agents interacting with each other.

    :ivar int N: Initial value: number of agents
    :ivar int m: Number of other agents to interact with
    :ivar float alpha: Controversialness of issue (sigmoidal shape)
    :ivar float K: Social interaction strength
    :ivar float epsilon: Minimum activity level with another agent
    :ivar float gamma: Power law distribution param
    :ivar float beta: Power law decay of connection probability
    :ivar p_mutual_interaction: Probability of a mutual interaction
    :ivar np.ndarray p_conn: Connection probabilities (matrix)
    :ivar np.ndarray activities: Activities of agents (vector)
    :ivar SocialInteraction adj_mat: Interactions of agents
    :ivar np.ndarray opinions: Opinions of agents (vector)
    :ivar SolverResult result: Post-simulation result with 't' time (vector) and 'y' opinion (agent x time) attrs

    """

    # noinspection PyTypeChecker
    def __init__(
        self,
        N: int,
        m: int,
        K: float,
        alpha: float,
        name="echochamber",
        seed=1337,
        *args,
        **kwargs,
    ):
        from opdynamics.dynamics.socialinteraction import SocialInteraction

        # create a random number generator for this object (to be thread-safe)
        self.rn = default_rng(seed)
        self._seed = seed

        # create a human-readable name for ths object
        self.name = name

        # assign args to object variables
        self.N = N
        self.m = m
        self.K = K
        self.alpha = alpha

        # private attributes
        self._dist: str = None
        self._beta: float = None

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
        self.result: SolverResult = None
        self.init_opinions()

    def init_opinions(self, min_val=-1.0, max_val=1.0):
        """Randomly initialise opinions for all N agents between [min_val, max_val] from a uniform distribution

        :param min_val: lowest value (inclusive)
        :param max_val: highest value (inclusive)
        """
        self.opinions = self.rn.uniform(min_val, max_val, size=self.N)
        self.result = None

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
        self._dist = f"{distribution.name}{dist_args}"

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
        self._beta = beta

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
        from opdynamics.dynamics.socialinteraction import SocialInteraction

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

    def set_dynamics(self, *args, **kwargs):
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

    @property
    def has_results(self):
        """Check if this object has a results property with simulation data."""
        return self.result is not None

    @property
    def current_time(self):
        return 0 if not self.has_results else self.result.t[-1]

    def _setup_run(
        self, dt: float = 0.01, t_end: float = 0.05
    ) -> Tuple[Tuple[float, float], tuple]:
        if (
            self.activities is None
            or self.p_conn is None
            or self.adj_mat is None
            or self.dy_dt is None
        ):
            raise ECSetupError

        args = (self.K, self.alpha, self.N, self.m, self.p_conn, self.adj_mat, dt)

        if not self.has_results:
            t_start = 0
            self.prev_result = None
        else:
            t_start = self.result.t[-1]
            t_end += t_start
            # noinspection PyTypeChecker
            self.prev_result: SolverResult = copy.deepcopy(self.result)
            logger.info(
                f"continuing dynamics from {t_start:.6f} until {t_end:.6f}. Opinions can be reset using "
                f"ec.init_opinions()."
            )
        return (np.round(t_start, 6), np.round(t_end, 6)), args

    def _post_run(self):
        # reassign opinions to last time point
        self.opinions = self.result.y[:, -1]
        if len(self.result.t) > 2 and self.result.t[0] > 0:
            self.result = self.prev_result + self.result
        logger.info(f"done running {self.name}")

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

        t_span, args = self._setup_run(dt, t_end)

        if method in ODE_INTEGRATORS:
            # use a custom method in `opdynamics.utils.integrators`
            self.result: SolverResult = solve_ode(
                self.dy_dt,
                t_span=t_span,
                y0=self.opinions,
                method=method,
                dt=dt,
                args=args,
                desc=self.name,
            )
        else:
            # use a method in `scipy.integrate`
            # use custom OdeResult (which SolverResult inherits from) for type
            from opdynamics.integrate.types import OdeResult

            # noinspection PyTypeChecker
            self.result = OdeResult(
                solve_ivp(
                    self.dy_dt,
                    t_span=t_span,
                    y0=self.opinions,
                    method=method,
                    vectorized=True,
                    args=args,
                    first_step=dt,
                    max_step=dt,
                )
            )
        self._post_run()

    def get_mean_opinion(
        self, t: Union[float, np.ndarray] = -1
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
        if isinstance(t, float):
            idx = np.argmin(np.abs(t - self.result.t))
        else:
            idx = t if t is not None else np.arange(len(self.result.t))
        time_point, average = self.result.t[idx], np.mean(self.result.y[:, idx], axis=0)
        return time_point, average

    def get_nearest_neighbours(self, t: int or float = -1) -> np.ndarray:
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
        idx = np.argmin(np.abs(t - self.result.t)) if isinstance(t, float) else t
        snapshot_adj_mat = self.adj_mat.accumulate(idx)
        out_degree_i = np.sum(snapshot_adj_mat, axis=0)
        close_opinions = np.sum(snapshot_adj_mat * self.result.y[:, idx], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            # suppress warnings about dividing by nan or 0
            nn = close_opinions / out_degree_i
        return nn

    def result_df(self):
        if self.result is None:
            raise RuntimeError(
                f"{self.name} has not been run. call `.run_network` first."
            )
        df = pd.DataFrame(self.result.y.T, index=self.result.t)
        df.name = "opinions"
        return df

    def get_change_in_opinions(self, dt=0.01):
        df = self.result_df()
        # create time array with constant dt (variable dt may have been used in the numerical integration method)
        t_indices = [
            np.nanargmin(np.abs(t - self.result.t))
            for t in np.arange(0, self.result.t[-1] + dt, dt)
        ]
        t = self.result.t[t_indices]
        opinions = self.result.y[:, t_indices]
        return np.diff(opinions, axis=1)

    def _get_filename(self):
        return os.path.join(".cache", f"{hash(self)}.h5")

    def save(self, only_last=True) -> str:
        """Save the echochamber to the cache using the HDF file format.

        File name and format specified in ``_get_filename()``

        :param only_last: Save only the last time point (default True).

        :return Saved filename.
        """
        import warnings
        from tables import NaturalNameWarning

        try:
            os.makedirs(".cache")
        except FileExistsError:
            pass
        filename = self._get_filename()

        df_opinions = self.result_df()
        _name = df_opinions.name
        # get dt
        _index_name = np.max(np.diff(df_opinions.index))
        if only_last:
            logger.debug("saving only last time point")
            # take last value but keep df_opinions as a DataFrame by including a `:`
            df_opinions = df_opinions.iloc[-1:]
            df_opinions.name = _name
        df_opinions.index.name = _index_name
        df_conn = pd.DataFrame(self.p_conn)
        df_conn.name = "p_conn"
        df_act = pd.Series(self.activities)
        df_act.name = "activities"
        df_adj_mat_accum = pd.DataFrame(self.adj_mat.accumulator)
        df_adj_mat_last = pd.DataFrame(self.adj_mat[-1])
        df_adj_mat_accum.name = f"adj_mat_accum-{self.adj_mat.p_mutual_interaction}"
        df_adj_mat_last.name = f"adj_mat_last-{self.adj_mat.p_mutual_interaction}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NaturalNameWarning)
            for df in [df_opinions, df_conn, df_act, df_adj_mat_accum, df_adj_mat_last]:
                df.to_hdf(filename, df.name)
        logger.debug(f"saved to {filename}\n{self}")
        return filename

    def load(self, dt, T):
        """
        Try to get previous results from the cache and return successful or not.
        :param dt: Time step accuracy of simulation to load. Asking for a coarser dt than a simulation has been run
            will load results (i.e. 0.01 when simulation was run with 0.001), but not the other way.
        :param T: Time point from which to retrieve results.
            Due to the way adj_mat is cached (only last adj_mat), only full simulations can be loaded.
            That is, asking for T=0.5 for a simulation that has run for T=1.0 will not load because the adjacency
            matrix at T=0.5 cannot be determined. A workaround is do short simulations and change the name of
            echochamber object between `run_network` calls.
        :return: True if loaded, False otherwise.
        """

        filename = self._get_filename()
        logger.debug(f"trying to hit cache for {filename}\n{self}")
        if os.path.exists(filename):
            dt_precision, dt_scale = precision_and_scale(dt)
            T_precision, T_scale = precision_and_scale(T)
            with pd.HDFStore(filename, mode="r") as hdf:
                keys = hdf.keys()
                # retrieve opinions for the time info first
                df = hdf.get("opinions")
                t_arr = df.index.values
                y_arr = df.values
                _dt = df.index.name
                _t_end = t_arr[-1]
                if (
                    np.round(T - _t_end, T_scale) != 0.0
                    or np.round(dt - _dt, dt_scale) != 0.0
                ):
                    return False
                self.result = SolverResult(
                    t_arr, y_arr.T, None, None, None, 0, 0, 0, 1, "success", True,
                )
                self._post_run()

                for key in keys:
                    df = hdf.get(key)
                    if "opinions" in key:
                        pass
                    elif "p_conn" in key:
                        self.p_conn = df.values
                    elif "activities" in key:
                        self.activities = df.values
                    elif "adj_mat" in key:
                        self.adj_mat.p_mutual_interaction = float(key.split("-")[-1])
                        if "accum" in key:
                            self.adj_mat._accumulator = df.values
                        elif "last" in key:
                            self.adj_mat._last_adj_mat = df.values
                        else:
                            raise KeyError(
                                f"unexpected adj_mat key '{key}' in hdf '{filename}'"
                            )
                    else:
                        raise KeyError(f"unexpected key '{key}' in hdf '{filename}'")

            logger.debug(f"{self.name} loaded from {filename}")
            return True
        return False

    def __repr__(self):
        return (
            f"{self.name}={self.__class__.__name__}(N={self.N},m={self.m},K={self.K},alpha={self.alpha},"
            f"seed={self._seed}) {self._dist} p_conn(beta={self._beta}) adj_mat(r={self.adj_mat.p_mutual_interaction})"
        )


class NoisyEchoChamber(EchoChamber):
    # noinspection PyTypeChecker
    def __init__(self, *args, name="noisy echochamber", **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.diffusion: diffeq = None
        self.wiener_process: Callable = None
        self._D_hist = []
        self.diff_args = ()

    def set_dynamics(self, D=0.01, noise_source=EXTERNAL_NOISE, *args, **kwargs):
        """Network with noise.

        External noise:
        -----

        .. math::
            \dot{x}_i = - x_i + K \cdot \sum_{j=1} A_{ij}(t) \cdot \\tanh{(\\alpha \cdot x_j)} + D \cdot \\xi (t)

        Where :math:`\\xi (t)` is the Wiener Process.

        Internal noise:
        -----

        .. math::
            \dot{x}_i = - x_i + K \cdot \sum_j A_{ij}(t) \cdot \\tanh{(\\alpha \cdot x_j)} + D \cdot (x_i - x_k)

        Where :math:`x_k` is an agent chosen every k time steps.

        :param D: Strength of noise.
        :param noise_source: Whether noise is external or internal (see formulations above).
            Use constants defined in `utils.constants`.

        :keyword k_steps: If `noise_source=INTERNAL_NOISE`, then choose N random agents every `k_steps`
            (default 10).

        # TODO: pick agent with opposite opinion

        """
        # assign drift as before, aka dy_dt
        super().set_dynamics()

        # create new diffusion term
        if noise_source >= INTERNAL_NOISE:
            self._idx = self.rn.uniform(0, self.N, self.N)
            self.diff_args = (kwargs.pop("k_steps", 10),)
            logger.debug(f"internal noise chosen with k_steps={self.diff_args}")
            precision, scale = precision_and_scale(self.diff_args[0])

            def choose_k(t, dt, _k_steps):
                if (scale and np.round(t, scale) % _k_steps == 0) or (
                    int(t / dt) % _k_steps == 0
                ):
                    self._idx = np.round(
                        self.rn.uniform(0, self.N - 1, size=self.N), 0
                    ).astype(int)

            def diffusion(t, y, *diff_args):
                choose_k(t, *diff_args)
                return D * (y - y[self._idx])

            def diffusion_tanh(t, y, *diff_args):
                choose_k(t, *diff_args)
                return D * np.tanh(y - y[self._idx])

            def diffusion_tanh_k(t, y, *diff_args):
                choose_k(t, *diff_args)
                return D * (y - np.tanh(y[self._idx]))

            if noise_source == INTERNAL_NOISE_SIG:
                self.diffusion = diffusion_tanh
            elif noise_source == INTERNAL_NOISE_SIG_K:
                self.diffusion = diffusion_tanh_k
            else:
                self.diffusion = diffusion
            self.wiener_process = lambda dt: 1
        else:
            self.diffusion = lambda t, y, *diff_args: D
            self.wiener_process = lambda dt: self.rn.normal(
                loc=0, scale=np.sqrt(dt), size=self.N
            )

        self._D_hist.append((self.current_time, D))

    def run_network(
        self, dt: float = 0.01, t_end: float = 0.05, method: str = "Euler-Maruyama",
    ):
        """Dynamics are no longer of an ordinary differential equation so we can't use scipy.solve_ivp anymore"""

        from opdynamics.integrate.solvers import SDE_INTEGRATORS, solve_sde

        t_span, args = self._setup_run(dt, t_end)
        diff_args = (dt, *self.diff_args)
        if method in SDE_INTEGRATORS:
            # use a custom method in `opdynamics.utils.integrators`
            self.result: SolverResult = solve_sde(
                self.dy_dt,
                self.diffusion,
                self.wiener_process,
                t_span=t_span,
                y0=self.opinions,
                method=method,
                dt=dt,
                args=args,
                diff_args=diff_args,
                desc=self.name,
            )
        else:
            raise NotImplementedError()
        self._post_run()

    def __repr__(self):
        d_hist = [f"D={_D:.5f} from {_t:.5f}" for _t, _D in self._D_hist]
        return f"{super().__repr__()} D_hist={d_hist} diff_args={self.diff_args}"


def example():
    """Simple example to show how to run a simulation and display some results."""
    from opdynamics.visualise.visechochamber import VisEchoChamber

    logging.basicConfig(level=logging.DEBUG)

    num_agents = 1000
    m = 10  # number of other agents to interact with
    alpha = 2  # controversialness of issue (sigmoidal shape)
    K = 3  # social interaction strength
    epsilon = 1e-2  # minimum activity level with another agent
    gamma = 2.1  # power law distribution param
    beta = 2  # power law decay of connection probability
    activity_distribution = negpowerlaw
    r = 0.5
    dt = 0.01
    t_end = 0.5

    ec = EchoChamber(num_agents, m, K, alpha, seed=1337)
    vis = VisEchoChamber(ec)

    ec.set_activities(activity_distribution, gamma, epsilon, 1)
    vis.show_activities()
    vis.show_activity_vs_opinion()

    ec.set_connection_probabilities(beta=beta)
    ec.set_social_interactions(r=r, dt=dt, t_end=t_end)
    ec.set_dynamics()

    ec.run_network(dt=dt, t_end=t_end)
    vis.show_opinions(color_code=False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    example()

    plt.show()
