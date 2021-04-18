"""

"""
import copy
from functools import lru_cache
from opdynamics.utils.plot_utils import get_time_point_idx
import os

import numpy as np
import pandas as pd
import logging

from typing import Callable, Tuple, Union

from numpy.random import default_rng
from pandas.errors import PerformanceWarning
from scipy.stats import powerlaw

from opdynamics.dynamics.opinions import dy_dt, sample_dy_dt
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
        self.dy_dt: diffeq = None
        self.result: SolverResult = None
        self.init_opinions()

        # other public attributes assigned during an operation
        self.save_txt: str = None

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

    def set_social_interactions(
        self,
        beta: float = 0.0,
        r: float = 0.5,
        store_all=False,
        dt: float = None,
        t_end: float = None,
        **kwargs,
    ):
        """Define the social interactions that occur at each time step.

        Populates `self.adj_mat` (adjacency matrix) that is used in opinion dynamics.

        For agent `i`, the probability of connecting to agent `j` is a function of the absolute strength of
        their opinions and a beta param, relative to all of the differences between an agent i and every other agent.

        .. math::
            p_{ij} = \\frac{|x_i - x_j|^{-\\beta}}{\sum_j |x_i - x_j|^{-\\beta}}

        :param beta: Power law decay of connection probability. Decay when beta>0, increase when beta<0.
            When beta=0, then connection probabilities are uniform.
        :param r: Probability of a mutual interaction [0,1].
        :param store_all: Store all the interaction matrice (True) or only the accumulative interactions and last interaction (False).
        :param dt: Time step being used in simulation. Specified here so interaction dynamics have a clear time step
            even if the integration of the opinion dynamics occurs at smaller time steps (e.g. with the RK method).
        :param t_end: Last time point. Together with dt, determines the size of the social interaction array.

        :keyword update_conn: Whether to update connection probabilities at every dt (default False).

        """
        from opdynamics.dynamics.socialinteraction import SocialInteraction

        if self.activities is None:
            raise RuntimeError(
                """Activities need to be set. 
                ec = EchoChamber(...)
                ec.set_activities(...)
                ec.set_connection_probabilities(...)
                ec.set_social_interactions(...)
                """
            )

        self.adj_mat = SocialInteraction(self, r, beta=beta, **kwargs)
        self._beta = beta
        if store_all:
            if t_end is None or dt is None:
                raise RuntimeError(
                    "`t_end` and `dt` need to be defined for pre-emptively storing adjacency matrix in "
                    "`set_social_interactions`"
                )
            self.adj_mat.store_interactions(t_end, dt)

    def set_dynamics(self, *args, **kwargs):
        """Set the dynamics of network by assigning a function to `self.dy_dt`.

        `self.dy_dt` is a function to be called by the ODE solver, which expects a signature of (t, y, *args).
        """

        self.dy_dt = dy_dt

    @property
    def has_results(self):
        """Check if this object has a results property with simulation data."""
        return self.result is not None

    @property
    def current_time(self):
        return 0 if not self.has_results else self.result.t[-1]

    @property
    @lru_cache(maxsize=1)
    def agent_idxs(self):
        return list(range(self.N))

    def _setup_run(self, t_end: float = 0.05) -> Tuple[float, float]:
        if self.activities is None or self.adj_mat is None or self.dy_dt is None:
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
        """Bundle parameters into a single tuple for ``dy_dt``.

        .. note ::

            ``dt`` will always be the last argument in ``dy_dt``

        .. note::
            For extending this method (along with ``dy_dt``) in subclasses, ``*args`` should be placed first so that
            ``parent_args`` can be collected:

            .. code-block :: python

                def dy_dt(t,y,*args):
                    new_arg, *parent_args = args
                    return parent_dy_dt(t, y, *parent_args) * new_arg

        """
        return (*args, self.K, self.alpha, self.adj_mat)

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
        # always have dt last
        args = (*self._args(), dt)

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
        self,
        sample_size: int,
        num_samples: int = 1,
        t: float = -1,
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
        t_idx = get_time_point_idx(self.result.t, t)
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
        return t_indices, t, np.diff(opinions, axis=1)

    @lru_cache(maxsize=None, typed=True)
    def get_network_graph(self, t: Union[Tuple[Union[int, float]], int, float] = -1):
        """Construct a graph of the network

        :param t: The time point (or range) for the network.

        :return: The graph object (from ``networkx``).
        """
        import networkx as nx
        from itertools import product

        t_idx = get_time_point_idx(self.result.t, t)
        if np.iterable(t_idx):
            last_t_idx = t_idx[1]
        else:
            last_t_idx = t_idx
            
        conn_weights = self.adj_mat.accumulate(t_idx)

        G = nx.DiGraph()

        df_opinions_at_t = self.result_df().iloc[last_t_idx]

        for i in self.agent_idxs:
            G.add_node(i, x=df_opinions_at_t[i])

        for i, j in product(*[range(N) for N in conn_weights.shape]):
            G.add_edge(i, j, weight=conn_weights[i, j])
            G.add_edge(j, i, weight=conn_weights[j, i])

        return G

    def get_network_agents(self, t: Union[Tuple[Union[int, float]], int, float] = -1):
        """Construct a graph of the network and determine the in-degree and out-degree of each agent in the network.

        The degree of an agent is the total number of connections it made with all other agents (directed if in- or out- are specified).

        :param t: See `EchoChamber.get_network_graph`

        :return: The graph object (from ``networkx``) and a dataframe with the degree information.
        """
        G = self.get_network_graph(t)
        # note that we explicitly provide the agent indices for an ordered result

        opinion = pd.Series(dict(G.nodes(data="x"))).loc[self.agent_idxs]

        # note that "degree" is the sum of in_degree and out_degree
        degree = pd.Series([d for n, d in G.degree(self.agent_idxs, weight="weight")])
        in_degree = pd.Series(
            [d for n, d in G.in_degree(self.agent_idxs, weight="weight")]
        )
        out_degree = pd.Series(
            [d for n, d in G.out_degree(self.agent_idxs, weight="weight")]
        )

        df_degree = pd.DataFrame(
            {
                "idx": self.agent_idxs,
                "opinion": opinion,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "degree": degree,
            }
        )

        return G, df_degree

    def get_network_connections(
        self, t: Union[Tuple[Union[int, float]], int, float] = -1, long=True
    ):
        """Construct a graph of the network and determine the interactions (connections) in the network.

        :param t: See `EchoChamber.get_network_graph`
        :param long: Whether the returned DataFrame should be in long (True) or wide (False) format. 
            See https://pandas.pydata.org/pandas-docs/stable/getting_started/intro_tutorials/07_reshape_table_layout.html#min-tut-07-reshape

        :return: The graph object (from ``networkx``) and a dataframe (long format by default) with the connection information.
        """
        G = self.get_network_graph(t)
        df_wide = pd.DataFrame(G.edges(data="weight"), columns=["A", "B", "weight"])
        if not long:
            return G, df_wide

        df_long = (
            df_wide[df_wide["weight"] != 0]
            .melt(id_vars=["A"], value_vars=["B"])
            .drop(columns=["variable"])
        )
        df_long.columns = ["A", "B"]

        return G, df_long

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
        from opdynamics.utils.cache import get_hash_filename

        return get_hash_filename(self)

    def save(
        self,
        only_last=True,
        complevel=DEFAULT_COMPRESSION_LEVEL,
        write_mapping=True,
        dt=None,
    ) -> str:
        """Save the echochamber to the cache using the HDF file format.

        File name and format specified in ``_get_filename()``

        :param only_last: Save only the last time point (default True).
        :param complevel: Compression level (default DEFAULT_COMPRESSION_LEVEL defined in ``constants``).
        :param write_mapping: Write to a file that maps the object's string representation and it's hash value.
        :param dt: Explicitly include the dt value for the index name. If not provided, it is calculated as the
            maximum time step from ``result_df``.
        
        :return: Saved filename.
        """
        import warnings
        from tables import NaturalNameWarning

        filename = self._get_filename()
        hash_txt = os.path.split(filename)[-1]

        df_opinions = self.result_df()
        _name = df_opinions.name
        # get dt
        _index_name = np.max(np.diff(df_opinions.index)) if dt is None else dt
        if only_last:
            logger.debug("saving only last time point")
            # take last value but keep df_opinions as a DataFrame by including a `:`
            df_opinions = df_opinions.iloc[-1:]
            df_opinions.name = _name
        df_opinions.index.name = _index_name
        df_act = pd.Series(self.activities)
        df_act.name = "activities"
        df_adj_mat_accum = pd.DataFrame(self.adj_mat.accumulator)
        df_adj_mat_last = pd.DataFrame(self.adj_mat[-1])
        df_adj_mat_accum.name = f"adj_mat_accum"
        df_adj_mat_last.name = f"adj_mat_last"
        meta = dict(complevel=complevel, complib="blosc:zstd")
        df_meta = pd.Series(meta, name="meta")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NaturalNameWarning)
            warnings.simplefilter("ignore", PerformanceWarning)
            for df in [
                df_opinions,
                df_act,
                df_adj_mat_accum,
                df_adj_mat_last,
                df_meta,
            ]:
                df.to_hdf(filename, df.name, complevel=7, complib="blosc:zstd")
        if self.adj_mat._time_mat is not None:
            adj_mat_file_compressed = self.adj_mat._time_mat.filename.replace(
                ".dat", ".npz"
            )
            logger.debug(f"saving full adj_mat to '{adj_mat_file_compressed}'")
            # save compressed version
            np.savez_compressed(
                adj_mat_file_compressed, time_mat=self.adj_mat._time_mat
            )
            new_time_mat = np.load(adj_mat_file_compressed, mmap_mode="r+")["time_mat"]
            # delete previous mmap file to explicitly clear storage
            del self.adj_mat._time_mat
            # link to stored version
            self.adj_mat._time_mat = new_time_mat
            logger.debug(f"...saved full adj_mat and deleted memory map")
        logger.debug(f"saved to {filename}\n{self}")
        self.save_txt = f"\n{self}\n\t{hash_txt}"
        if write_mapping:
            print("write to file")
            with open(
                os.path.join(os.path.split(filename)[0], "map.txt"), "a+"
            ) as f_map:
                f_map.write(self.save_txt)

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

            cached_results = {
                "opinions": False,
                "activities": False,
                "adj_mat_accum": False,
                "adj_mat_last": False,
            }
            loaded_keys = copy.deepcopy(cached_results)

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
                compressed = False
                cached_results["opinions"] = SolverResult(
                    t_arr,
                    y_arr.T,
                    None,
                    None,
                    None,
                    0,
                    0,
                    0,
                    1,
                    "success",
                    True,
                )
                loaded_keys["opinions"] = True
                for key in keys:
                    df: Union[pd.DataFrame, pd.Series, object] = hdf.get(key)
                    if key.startswith("/"):
                        key = key[1:]
                    if "-" in key:
                        # backwards compatible
                        key, p = key.split("-")
                    if "opinions" in key:
                        continue
                    elif "meta" in key:
                        if df.loc["complevel"] > 0:
                            compressed = True
                    cached_results[key] = df.values
                    loaded_keys[key] = True

                if all(loaded_keys.values()):
                    self.result = cached_results["opinions"]
                    self.activities = cached_results["activities"]
                    self.adj_mat._accumulator = cached_results["adj_mat_accum"]
                    self.adj_mat._last_adj_mat = cached_results["adj_mat_last"]
                    self._post_run()
                else:
                    # not everything loaded
                    return False

            if self.adj_mat._time_mat is not None:
                adj_mat_file_compressed = self.adj_mat._time_mat.filename.replace(
                    ".dat", ".npz"
                )

                new_time_mat = np.load(adj_mat_file_compressed, mmap_mode="r+")[
                    "time_mat"
                ]
                # delete previous mmap file to explicitly clear storage
                del self.adj_mat._time_mat
                #
                self.adj_mat._time_mat = new_time_mat

            logger.debug(f"{self.name} loaded from {filename}")
            if not compressed:
                self.save(only_last=self.result.y.shape[1] > 1, dt=_dt)
            return True
        return False

    def __repr__(self):
        return (
            f"{self.name}={self.__class__.__name__}(N={self.N},m={self.m},K={self.K},alpha={self.alpha},"
            f"seed={self._seed}) {self._dist} {self.adj_mat}"
        )


class ConnChamber(EchoChamber):
    """Network that calculates new connection probabilities at every time step, optionally specifying the probability,
    ``p_opp``, that an agent will interact with another agent holding an opposing opinion."""

    def set_social_interactions(self, *args, p_opp=0, update_conn=True, **kwargs):
        from opdynamics.dynamics.socialinteraction import (
            get_connection_probabilities_opp,
        )

        conn_method = kwargs.pop("conn_method", get_connection_probabilities_opp)
        super().set_social_interactions(
            *args,
            conn_method=conn_method,
            update_conn=update_conn,
            p_opp=p_opp,
            **kwargs,
        )


class NoisyEchoChamber(EchoChamber):
    def __init__(self, *args, name="noisy echochamber", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self._D_hist = []
        self.D: float = 0
        self.super_dy_dt: Callable = None

    def set_dynamics(self, D=0.01, *args, **kwargs):
        """Network with noise.

        :param D: Strength of noise.

        """
        super().set_dynamics(*args, **kwargs)
        self._D_hist.append((self.current_time, D))
        self.D = D

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
        self,
        dt: float = 0.01,
        t_end: float = 0.05,
        method: str = "Euler-Maruyama",
    ):
        """Dynamics are no longer of an ordinary differential equation so we can't use scipy.solve_ivp anymore"""

        from opdynamics.integrate.solvers import SDE_INTEGRATORS, solve_sde

        t_span = self._setup_run(t_end)
        args = (*self._args(), dt)
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
        return super()._args(*args, self.k_steps, self.alpha_2)


class SampleChamber(NoisyEchoChamber):
    """
    Provide a mean sample of opinions to each agent.

    see https://en.wikipedia.org/wiki/Central_limit_theorem
    """

    # noinspection PyTypeChecker
    def __init__(self, *args, name="sample chamber", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self._sample_size: int = 0
        self._sample_means: float = 0.0
        self._sample_method: (str, Callable) = None

    def set_dynamics(
        self,
        D: float = 0,
        sample_size: int = 20,
        sample_method: str = "basic",
        *args,
        **kwargs,
    ):
        """Set the dynamics of network by assigning a function to `self.dy_dt`."""
        from opdynamics.dynamics.opinions import clt_methods

        super().set_dynamics(D, *args, **kwargs)
        self.super_dy_dt = self.dy_dt

        # object to store sample_means value at distinct time points
        self._sample_means = 0
        self._sample_size = sample_size
        self.dy_dt = sample_dy_dt
        if type(sample_method) is str:
            assert (
                sample_method in clt_methods
            ), f"sample_method must be one of '{clt_methods.keys()}'"
            self._sample_method = (sample_method, clt_methods[sample_method])
        else:
            try:
                self._sample_method = (sample_method.__name__, sample_method)
            except KeyError:
                logger.error(f"`sample_method` must be a `str` or callable function")

    def _args(self, *args):
        return super()._args(
            *args, self._sample_method[1], self, self._sample_size, self.N
        )

    def __repr__(self):
        return f"{super().__repr__()} sample_method={self._sample_method[0]} n={self._sample_size}"


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
