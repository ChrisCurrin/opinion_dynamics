""""""
import copy
import os

import numpy as np
import pandas as pd
import logging

from typing import Callable, Tuple, Union

from numpy.random import default_rng
from scipy.stats import powerlaw

from opdynamics.metrics.opinions import (
    distribution_modality,
    nearest_neighbours,
    sample_means,
)
from opdynamics.utils.accuracy import precision_and_scale
from opdynamics.utils.constants import DEFAULT_COMPRESSION_LEVEL
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

    def set_activities(
        self, distribution=negpowerlaw, *dist_args, dim: int = 1, **kwargs
    ):
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

    def set_connection_probabilities(self, beta: float = 0.0, **kwargs):
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
            # get activity matrix for this time point
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

    def _setup_run(self, t_end: float = 0.05) -> Tuple[float, float]:
        if (
            self.activities is None
            or self.p_conn is None
            or self.adj_mat is None
            or self.dy_dt is None
        ):
            raise ECSetupError

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
        return np.round(t_start, 6), np.round(t_end, 6)

    def _args(self, *args):
        return (self.K, self.alpha, self.N, self.m, self.p_conn, self.adj_mat, *args)

    def _post_run(self):
        # reassign opinions to last time point
        self.opinions = self.result.y[:, -1]
        if len(self.result.t) > 2 and self.result.t[0] > 0:
            self.result = self.prev_result + self.result
        logger.info(f"done running {self.name}")

    def run_network(
        self, dt: float = 0.01, t_end: float = 0.05, method: str = "Euler"
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

        t_span = self._setup_run(t_end)
        args = self._args(dt)

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

    def get_sample_means(
        self, sample_size: int, num_samples: int = 1, t: float = -1,
    ) -> np.ndarray:
        """
        Calculate the sample means.

        Each mean is from a sample of ``sample_size`` agents.

        The number of means is the same as ``num_samples``.

        Means are taken either from ``opinions`` argument or from ``self.result.y`` at time point ``t``.

        see https://en.wikipedia.org/wiki/Central_limit_theorem

        :param sample_size: Pick this many agents' opinions (i.e. a sample).
        :param num_samples: Number of sample to perform.
        :param t: Time at which to conduct the sampling (-1 for last time point).
        :return: Array of means (size equal to ``num_samples``).
        """

        t_idx = np.argmin(np.abs(t - self.result.t)) if isinstance(t, float) else t
        opinions = self.result.y[:, t_idx]
        return sample_means(opinions, sample_size, num_samples, rng=self.rn)

    def get_nearest_neighbours(self, t: Union[int, float] = -1) -> np.ndarray:
        """Calculate mean value of every agents' nearest neighbour.

        .. math::
                \\frac{\\sum_j a_{ij} x_j}{\\sum_j a_{ij}}

        where :math:`a_{ij}` represents the (static) adjacency matrix of the aggregated interaction network
        and :math:`\\sum_j a_{ij}` is the degree of node `i`.

        """
        idx = np.argmin(np.abs(t - self.result.t)) if isinstance(t, float) else t
        snapshot_adj_mat = self.adj_mat.accumulate(idx)
        opinions = self.result.y[:, idx]
        return nearest_neighbours(opinions, snapshot_adj_mat)

    # noinspection NonAsciiCharacters
    def get_distribution_modality(self, t: Union[int, float] = -1) -> float:
        """Calculate Test of unimodality for normal distribution(s)

        .. math::
                \\frac{v - \\mu}{\\sigma} \\leq \\sqrt{\\frac{3}{5}}

        where :math:`v` is the median, :math:`\\mu`` is the mean, and :math:`\\sigma` is the standard deviation.

        see https://en.wikipedia.org/wiki/Unimodality
        see https://doi.org/10.1007/s10182-008-0057-2 

        """
        t_idx = np.argmin(np.abs(t - self.result.t)) if isinstance(t, float) else t
        opinions = self.result.y[:, t_idx]
        return distribution_modality(opinions)

    def get_change_in_opinions(self, dt=0.01) -> np.ndarray:
        """Calculate the change of opinions for a given time step ``dt``."""
        # create time array with constant dt (variable dt may have been used in the numerical integration method)
        t_indices = [
            np.nanargmin(np.abs(t - self.result.t))
            for t in np.arange(0, self.result.t[-1] + dt, dt)
        ]
        t = self.result.t[t_indices]
        opinions = self.result.y[:, t_indices]
        return np.diff(opinions, axis=1)

    def result_df(self) -> pd.DataFrame:
        """Provide opinions as a pandas ``DataFrame``"""
        if self.result is None:
            raise RuntimeError(
                f"{self.name} has not been run. call `.run_network` first."
            )
        df = pd.DataFrame(self.result.y.T, index=self.result.t)
        df.name = "opinions"
        return df

    def _get_filename(self) -> str:
        """get a cacheable filename for this instance"""
        from opdynamics.utils.cache import get_cache_dir

        cache_dir = get_cache_dir()
        return os.path.join(cache_dir, f"{hash(self)}.h5")

    def save(self, only_last=True, complevel=DEFAULT_COMPRESSION_LEVEL) -> str:
        """Save the echochamber to the cache using the HDF file format.

        File name and format specified in ``_get_filename()``

        :param only_last: Save only the last time point (default True).
        :param complevel: Compression level (default DEFAULT_COMPRESSION_LEVEL defined in ``constants``).

        :return Saved filename.
        """
        import warnings
        from tables import NaturalNameWarning

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
        meta = dict(complevel=complevel, complib="blosc:zstd")
        df_meta = pd.Series(meta, name="meta")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NaturalNameWarning)
            for df in [
                df_opinions,
                df_conn,
                df_act,
                df_adj_mat_accum,
                df_adj_mat_last,
                df_meta,
            ]:
                df.to_hdf(filename, df.name, complevel=7, complib="blosc:zstd")
        logger.debug(f"saved to {filename}\n{self}")
        with open(os.path.join(os.path.split(filename)[0], "map.txt"), "a+") as f_map:
            f_map.write(f"\n{self}\n\t{os.path.split(filename)[-1]}")
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
                compressed = False
                for key in keys:
                    df: Union[pd.DataFrame, pd.Series, object] = hdf.get(key)
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
                    elif "meta" in key:
                        if df.loc["complevel"] > 0:
                            compressed = True
                    else:
                        raise KeyError(f"unexpected key '{key}' in hdf '{filename}'")

            logger.debug(f"{self.name} loaded from {filename}")
            if not compressed:
                self.save(only_last=self.result.y.shape[1] > 1)
            return True
        return False

    def __repr__(self):
        return (
            f"{self.name}={self.__class__.__name__}(N={self.N},m={self.m},K={self.K},alpha={self.alpha},"
            f"seed={self._seed}) {self._dist} p_conn(beta={self._beta}) adj_mat(r={self.adj_mat.p_mutual_interaction})"
        )


class ConnChamber(EchoChamber):
    """Network that calculates new connection probabilities at every time step, optionally specifying the probability,
     ``p_opp``, that an agent will interact with another agent holding an opposing opinion."""

    def set_connection_probabilities(self, beta=0.0, p_opp=0.0, **kwargs):
        """For agent `i`, the probability of connecting to agent `j` is a function of the absolute strength of
            their opinions and a beta param, relative to all of the differences between an agent i and every
            other agent.

            .. math::
                p_{ij} = \\frac{|x_i - x_j|^{-\\beta}}{\sum_j |x_i - x_j|^{-\\beta}}

            :param beta: Power law decay of connection probability. Decay when beta>0, increase when beta<0.
                When beta=0, then connection probabilities are uniform.
            :param p_opp: Probability of inverting the power-law from decay to gain, making it more likely to connect
                with an agent holding an opposing opinion and less likely to connect with an agent holding a similar
                opinion.

        """

        p_conn = np.zeros(shape=(self.N, self.N))
        if p_opp > 0:
            betas = self.rn.choice([beta, -beta], size=self.N, p=[p_opp, 1 - p_opp])
        else:
            betas = -beta
        for i in range(self.N):
            mag = np.abs(self.opinions[i] - self.opinions)
            mag[i] = np.nan
            p_conn[i] = np.power(mag, betas)
            p_conn[i, i] = 0
            p_conn[i] /= np.sum(p_conn[i])
        self.p_conn = p_conn
        self._beta = beta
        self.p_opp = p_opp

    def set_social_interactions(self, *args, **kwargs):
        """Same as original, except the dynamics of re-calculating connection probabilities only works when
        lazy=True."""
        kwargs["lazy"] = True
        super().set_social_interactions(*args, **kwargs)

    def set_dynamics(self, *args, **kwargs):
        """Set the dynamics of network by assigning a function to `self.dy_dt`.

        `self.dy_dt` is a function to be called by the ODE solver, which expects a signature of (t, y, *args).
        """

        def dy_dt(t: float, y: np.ndarray, *args) -> np.ndarray:
            """Activity-Driven (AD) network dynamics.

            1. calculate connection probabilities based on difference in opinions
            2. get the interactions (A) that happen at this time point between each of N agents based on activity
            probabilities (p_conn) and the number of agents to interact with (m).
            3. calculate opinion derivative by getting the scaled (by social influence, alpha) opinions (y.T) of agents
            interacting with each other (A), multiplied by social interaction strength (K).

            """
            K, alpha, N, m, p_conn, A, beta, p_opp, dt = args
            # recalculate connection probabilities to be used in A
            p_conn(beta, p_opp)
            # get activity matrix for this time point
            At = A[int(t / dt)]
            return -y.T + K * np.sum(At * np.tanh(alpha * y.T), axis=1)

        self.dy_dt = dy_dt

    def _args(self, *args):
        self.p_conn = self.set_connection_probabilities
        return super()._args(self._beta, self.p_opp, *args)


class NoisyEchoChamber(EchoChamber):
    def __init__(self, *args, name="noisy echochamber", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self._D_hist = []

    def set_dynamics(self, D=0.01, *args, **kwargs):
        """Network with noise.

        :param D: Strength of noise.

        """
        super().set_dynamics(*args, **kwargs)
        self._D_hist.append((self.current_time, D))

    def __repr__(self):
        d_hist = [f"D={_D:.5f} from {_t:.5f}" for _t, _D in self._D_hist]
        return f"{super().__repr__()} D_hist={d_hist}"


class OpenChamber(NoisyEchoChamber):
    # noinspection PyTypeChecker
    def __init__(self, *args, name="open echochamber", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.diffusion: diffeq = None
        self.wiener_process: Callable = None
        self.diff_args = ()

    def set_dynamics(self, D=0.01, *args, **kwargs):
        """Network with noise.

        External noise:
        -----

        .. math::
            \\dot{x}_i = K \\cdot \\sum_{j=1} A_{ij}(t) \\cdot \\tanh{(\\alpha \\cdot x_j)} + D \\cdot \\xi (t)

        Where :math:`\\xi (t)` is the Wiener Process.

        :param D: Strength of noise.

        """
        # assign drift as before, aka dy_dt
        super().set_dynamics(D=D)

        self.diffusion = lambda t, y, *diff_args: D
        self.wiener_process = lambda dt: self.rn.normal(
            loc=0, scale=np.sqrt(dt), size=self.N
        )

    def run_network(
        self, dt: float = 0.01, t_end: float = 0.05, method: str = "Euler-Maruyama",
    ):
        """Dynamics are no longer of an ordinary differential equation so we can't use scipy.solve_ivp anymore"""

        from opdynamics.integrate.solvers import SDE_INTEGRATORS, solve_sde

        t_span = self._setup_run(t_end)
        args = self._args(dt)
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
        return f"{super().__repr__()} diff_args={self.diff_args}"


class ContrastChamber(NoisyEchoChamber):
    # noinspection PyTypeChecker
    def __init__(self, *args, name="contrast echochamber", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.k_steps: int = None
        self.alpha_2: float = None

    def set_dynamics(self, D=0.01, k_steps=1, alpha_2=1.0, contrast=False, **kwargs):
        """

        .. math::
            \\dot{x}_i = K \\cdot \\sum_j A_{ij}(t) \\cdot \\tanh{(\\alpha \\cdot x_j)} +
            D \\cdot \\tanh(\\alpha_2 \\cdot (x_i - x_k))

        .. math::
            \\dot{x}_i = K \\cdot \\sum_j A_{ij}(t) \\cdot \\tanh{(\\alpha \\cdot x_j)} +
            D \\cdot \\tanh(\\alpha_2 \\cdot x_k)

        Where :math:`x_k` is an agent chosen every `k` time steps.

        :param D: Strength of noise.
        :param k_steps: Choose N random agents every `k` time steps (default 1).
        :param alpha_2: Scaling of non-linearity
        :param contrast: Whether to nudge using only agent k's opinion (False),
            or the difference between the current agent and agent k (True).
        """
        # assign drift as before, aka dy_dt
        super().set_dynamics(D=D)
        super_dy_dt = self.dy_dt

        self._idx = self.rn.uniform(0, self.N, self.N)
        precision, scale = precision_and_scale(k_steps)

        def choose_k(t, dt, _k_steps):
            if (scale and np.round(t, scale) % _k_steps == 0) or (
                int(t / dt) % _k_steps == 0
            ):
                self._idx = np.round(
                    self.rn.uniform(0, self.N - 1, size=self.N), 0
                ).astype(int)

        def contrast_k(t, y, *nudge_args):
            _k_steps, _alpha_2, dt = nudge_args
            choose_k(t, dt, _k_steps)
            return D * np.tanh(_alpha_2 * (y - y[self._idx]))

        def show_k(t, y, *nudge_args):
            _k_steps, _alpha_2, dt = nudge_args
            choose_k(t, dt, _k_steps)
            return D * np.tanh(_alpha_2 * y[self._idx])

        nudge = contrast_k if contrast else show_k

        def dy_dt(t, y, *all_args):
            nudge_args, *args = all_args
            dt = args[-1]
            return super_dy_dt(t, y, *args) + nudge(t, y, *(*nudge_args, dt))

        self.dy_dt = dy_dt
        self.k_steps = k_steps
        self.alpha_2 = alpha_2

    def _args(self, *args):
        temp = super()._args(*args)
        return (self.k_steps, self.alpha_2, *temp)


class SampleChamber(NoisyEchoChamber):
    """
    Provide a mean sample of opinions to each agent.

    see https://en.wikipedia.org/wiki/Central_limit_theorem
    """

    # noinspection PyTypeChecker
    def __init__(self, *args, name="sample chamber", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.sample_size: int = None

    def set_dynamics(self, D=0, sample_size=20, *args, **kwargs):
        """Set the dynamics of network by assigning a function to `self.dy_dt`.
        """
        super().set_dynamics(D, *args, **kwargs)
        super_dy_dt = self.dy_dt

        def dy_dt(t: float, y: np.ndarray, *all_args) -> np.ndarray:
            """Activity-Driven (AD) network dynamics.

            1 - 3 as in ``ConnChamber``

            4. add a "population opinion" term that captures the Lindeberg–Lévy Central Limit Theorem -
            :math:`\\sqrt {n}\\left({\\bar{X}}_{n}-\\mu \\right) \\rightarrow \mathcal{N}\\left(0,\\sigma ^{2}\\right)`
            \\
            where :math:`X` is a random sample and :math:`\\bar{X}_{n}` is the sample mean for :math:`n` random samples.

            """
            n, N, *other_args = all_args
            if type(n) is tuple:
                # choose between low and high values (randint not implemented for default_rng)
                n = self.rn.choice(np.arange(n[0], n[1], dtype=int))
            return super_dy_dt(t, y, *other_args) + D * np.sqrt(n) * (
                sample_means(y, n, num_samples=N, rng=self.rn) - np.mean(y)
            )

        self.sample_size = sample_size
        self.dy_dt = dy_dt

    def _args(self, *args):
        temp = super()._args(*args)
        return (self.sample_size, self.N, *temp)


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

    ec = SampleChamber(num_agents, m, K, alpha, seed=1337)
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
