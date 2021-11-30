"""
# Networks of agents that give rise to echo chambers. 

"""
import copy
import logging
import os
from functools import lru_cache, partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas.errors import PerformanceWarning
from scipy.stats import powerlaw

from opdynamics.dynamics.opinions import dy_dt, sample_dy_dt, sample_dy_dt_activity
from opdynamics.integrate.types import SolverResult, diffeq
from opdynamics.metrics.opinions import (
    distribution_modality,
    nearest_neighbours,
    sample_means,
)
from opdynamics.utils.accuracy import precision_and_scale
from opdynamics.utils.constants import DEFAULT_COMPRESSION_LEVEL
from opdynamics.utils.decorators import hashable
from opdynamics.utils.distributions import negpowerlaw
from opdynamics.utils.errors import ECSetupError
from opdynamics.utils.plot_utils import get_time_point_idx

logger = logging.getLogger("social_networks")


@hashable
class SocialNetwork(object):
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
        name="SocialNetwork",
        seed=1337,
        filename: str = None,
        *args,
        **kwargs,
    ):
        from opdynamics.dynamics.socialinteractions import SocialInteraction

        self.seed = seed

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
        self._store_interactions: bool = False

        # quick checks
        assert N > 0 and type(N) is int
        assert 0 < m < N and type(m) is int
        assert alpha >= 0
        assert K >= 0

        # create array variables
        self.adj_mat: SocialInteraction = None
        self.activities: np.ndarray = None
        self.dy_dt: diffeq = None
        self.result: SolverResult = None
        self.init_opinions()

        # other public attributes assigned during an operation
        self.save_txt: str = None
        self.filename: str = filename

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value: int):
        # create a random number generator for this object (to be thread-safe)
        self.rn = default_rng(value)
        self._seed = value

    def init_opinions(self, min_val=-1.0, max_val=1.0):
        """Randomly initialise opinions for all N agents between [min_val, max_val] from a uniform distribution

        :param min_val: lowest value (inclusive)
        :param max_val: highest value (inclusive)
        """
        # create a dummy result in case `opinions` property is requested, which requires `result`
        self.result = SolverResult(
            np.array([0]),
            np.expand_dims(self.rn.uniform(min_val, max_val, size=self.N), 1),
            None,
            None,
            None,
            0,
            0,
            0,
            1,
            "init",
            True,
        )

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

        """
        from opdynamics.dynamics.socialinteractions import SocialInteraction

        if self.activities is None:
            raise RuntimeError(
                """Activities need to be set. 
                sn = SocialNetwork(...)
                sn.set_activities(...)
                sn.set_connection_probabilities(...)
                sn.set_social_interactions(...)
                """
            )
        self.adj_mat = SocialInteraction(self, r, beta=beta, **kwargs)
        self._beta = beta
        self._store_interactions = store_all

    def set_dynamics(self, *args, **kwargs):
        """Set the dynamics of network by assigning a function to `self.dy_dt`.

        `self.dy_dt` is a function to be called by the ODE solver, which expects a signature of (t, y, *args).
        """

        self.dy_dt = dy_dt

    def opinions_at_t(
        self, t: Union[Tuple[Union[int, float]], int, float] = -1, flatten: int = True
    ):
        t_idx = get_time_point_idx(self.result.t, t)
        _opinions: np.ndarray = self.result.y[:, t_idx]
        if flatten:
            _opinions = _opinions.ravel()
        return t_idx, _opinions

    @property
    def opinions(self) -> np.ndarray:
        return self.opinions_at_t(-1)[-1]

    @property
    def all_opinions(self) -> np.ndarray:
        return self.result.y

    @property
    def current_time(self):
        return self.result.t[-1]

    @property
    @lru_cache(maxsize=1)
    def agent_idxs(self):
        return list(range(self.N))

    @property
    def filename(self) -> str:
        """get a cacheable filename for this instance"""
        from opdynamics.utils.cache import get_hash_filename

        if self._filename is None:
            return get_hash_filename(self)
        return self._filename

    @filename.setter
    def filename(self, value: Optional[str]):
        if value is not None:
            from opdynamics.utils.cache import get_cache_dir

            if os.path.split(value)[0] == "":
                # no parent specified
                value = os.path.join(get_cache_dir(), value)
            root, ext = os.path.splitext(value)
            value = value + ".h5"

            if self.adj_mat is not None:
                self.adj_mat.filename = value

        self._filename = value

    def _setup_run(self, dt: float, t_dur: float) -> Tuple[float, float]:
        if self.activities is None or self.adj_mat is None or self.dy_dt is None:
            raise ECSetupError

        t_end = t_dur
        t_start = self.result.t[-1]
        t_end += t_start
        if t_start == 0:
            self.prev_result = None
        else:
            # noinspection PyTypeChecker
            self.prev_result: SolverResult = copy.deepcopy(self.result)
        logger.debug(f"running dynamics from {t_start:.6f} until {t_end:.6f}. dt={dt}")

        if self._store_interactions:
            self.adj_mat.store_interactions(dt, t_end)
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
        if len(self.result.t) > 2 and self.result.t[0] > 0:
            # concatenate results
            self.result = self.prev_result + self.result
        logger.debug(f"done running {self.name}")

    def run_network(
        self,
        dt: float = 0.01,
        t_dur: float = 0.05,
        method: str = "Euler",
        show_method_pbar: bool = True,
    ) -> None:
        """Run a simulation for the SocialNetwork until `t_dur` with a time step of `dt`.

        Because the SocialNetwork has ODE dynamics, an appropriate method should be chosen from
        `scipy.integrate.solver_ivp` or `opdynamics.integrate.solvers`

        :param dt: (Max) Time step for integrator. Smaller values will yield more accurate results but the simulation
            will take longer. Large `dt` for unstable methods (like "Euler") can cause numerical instability where
            results show **increasingly large** oscillations in opinions (nonsensical).
        :param t_dur: Time for simulation to span. Number of iterations will be at least t_dur/dt.
        :param method: Integration method to use. Must be one specified by `scipy.integrate.solver_ivp` or
            `opdynamics.integrate.solvers`
        :param show_method_pbar: Show a progress bar for the integration method, if custom.

        """
        from scipy.integrate import solve_ivp

        from opdynamics.integrate.solvers import ODE_INTEGRATORS, solve_ode

        t_span = self._setup_run(dt, t_dur)
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
                show_pbar=show_method_pbar,
            )
        else:
            # use a method in `scipy.integrate`
            # use custom OdeResult (which SolverResult inherits from) for type
            from opdynamics.integrate.types import OdeResult
            # TODO: add t_eval to solve_ivp
            t_eval = np.arange(t_span[0], t_span[1], dt)
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
        idx, opinions = self.opinions_at_t(t)
        snapshot_adj_mat = self.adj_mat.accumulate(idx)
        return nearest_neighbours(opinions, snapshot_adj_mat)

    # noinspection NonAsciiCharacters
    def get_distribution_modality(self, t: Union[int, float] = -1) -> float:
        """Calculate distance of peaks

        .. math::
               \\Lambda_x = \\argmax_{x>0} \\frac{f}{w} - \\argmax_{x<0} \\frac{f}{w}

        Where :math:`f` is the frequency of opinions in a bin width of :math:`w.

        :math:`w` was determined from the minimum of the Sturges and Freedman-Diaconis bin estimation methods:

        .. math::
                w = \\min ( \\frac{\\max x_R - \\min x_R}{\\log_{2}N + 1}, 2 \\frac{\\rm{IQR}}{N^\\frac{1}{3}} )

        where :math:`x_R` is an opinion subset (:math:`x>0` or :math:`x<0`) and :math:`\\rm{IQR}` is the :math:`x_R` interquartile range.

        The peak distance :math:`\Lambda_x` can be intuitively understood as the degree of polarization of echo chambers.

        For :math:`\Lambda_x` close to 0, the distribution of opinions is normal and depolarized, and for larger :math:`\Lambda_x`,
         the opinions of agents are polarized.


        """
        t_idx, opinions = self.opinions_at_t(t)
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
        from itertools import product

        import networkx as nx

        t_idx, opinions = self.opinions_at_t(t)

        conn_weights = self.adj_mat.accumulate(t_idx)

        G = nx.DiGraph()

        if np.iterable(t_idx):
            # just need last time index for last opinion
            t_idx = t_idx[-1]

        df_opinions_at_t = self.result_df().iloc[t_idx]

        for i in self.agent_idxs:
            G.add_node(i, x=df_opinions_at_t[i])

        for i, j in product(*[range(N) for N in conn_weights.shape]):
            G.add_edge(i, j, weight=conn_weights[i, j])
            G.add_edge(j, i, weight=conn_weights[j, i])

        return G

    def get_network_agents(self, t: Union[Tuple[Union[int, float]], int, float] = -1):
        """Construct a graph of the network and determine the in-degree and out-degree of each agent in the network.

        The degree of an agent is the total number of connections it made with all other agents (directed if in- or out- are specified).

        :param t: See `SocialNetwork.get_network_graph`

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

        :param t: See `SocialNetwork.get_network_graph`
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

    def save(
        self,
        only_last=True,
        complevel=DEFAULT_COMPRESSION_LEVEL,
        write_mapping=True,
        dt=None,
        raise_error=True,
    ) -> str:
        """Save the SocialNetwork to the cache using the HDF file format.

        File name and format specified in ``filename``

        :param only_last: Save only the last time point (default True).
        :param complevel: Compression level (default DEFAULT_COMPRESSION_LEVEL defined in ``constants``).
        :param write_mapping: Write to a file that maps the object's string representation and it's hash value.
        :param dt: Explicitly include the dt value for the index name. If not provided, it is calculated as the
            maximum time step from ``result_df``.

        :return: Saved filename.
        """
        import warnings

        from tables import NaturalNameWarning
        from tables.exceptions import HDF5ExtError

        logger.debug(f"saving {self}")

        filename = self.filename
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

        try:

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
                    df.to_hdf(filename, df.name, **meta)
            # only saves if self._store_interactions is True
            self.adj_mat.compress()
        except (HDF5ExtError, AttributeError) as err:
            err_msg = f"Could not save {self} to {filename}. \n{err}"
            if raise_error:
                logger.error(err_msg)
                print(err_msg)
                raise err
            else:
                logger.warning(err_msg)
        else:
            logger.debug(f"saved\n{self}\n-> {filename}")
            self.save_txt = f"\n{self}\n\t{hash_txt}"
            if write_mapping:
                if isinstance(write_mapping, str):
                    map_file_name = write_mapping
                else:
                    map_file_name = os.path.join(os.path.split(filename)[0], "map.txt")
                logger.debug(f"write to '{map_file_name}'")
                with open(map_file_name, "a+") as f_map:
                    f_map.write(self.save_txt)

        return filename

    def load(self, dt, T, raise_error=False):
        """
        Try to get previous results from the cache and return successful or not.
        :param dt: Time step accuracy of simulation to load. Asking for a coarser dt than a simulation has been run
            will load results (i.e. 0.01 when simulation was run with 0.001), but not the other way.
        :param T: Time point from which to retrieve results.
            Due to the way adj_mat is cached (only last adj_mat), only full simulations can be loaded.
            That is, asking for T=0.5 for a simulation that has run for T=1.0 will not load because the adjacency
            matrix at T=0.5 cannot be determined. A workaround is do short simulations and change the name of
            SocialNetwork object between `run_network` calls.
        :param raise_error: Raise an error if the file exists but not loaded properly, otherwise return False.
        :return: True if loaded, False otherwise.
        """
        from tables.exceptions import HDF5ExtError

        filename = self.filename
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

            try:
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

                adj_mat_file_compressed = self.adj_mat.filename.replace(
                        ".dat", ".npz"
                    )
                if os.path.exists(adj_mat_file_compressed):
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
            except (HDF5ExtError, AttributeError) as err:
                os.remove(filename)
                if raise_error:
                    raise err
                else:
                    logger.warning(f"{self} failed to load from {filename}\n{err}")
        return False

    def __repr__(self):
        return (
            f"{self.name}={self.__class__.__name__}(N={self.N},m={self.m},K={self.K},alpha={self.alpha},"
            f"seed={self._seed}) {self._dist} {self.adj_mat}"
        )


class ConnChamber(SocialNetwork):
    """Network that calculates new connection probabilities at every time step, optionally specifying the probability,
    ``p_opp``, that an agent will interact with another agent holding an opposing opinion.

    Note that this can be trivially used with NoisySocialNetworks (below) by passing ``conn_method``, ``p_opp``, to the
    :meth:`set_social_interactions` method.

    This class demonstrates a default implementation of continuous connection updates. Updating the connections at every time step doesn't
    seem to change the results but does incur additional computational costs.

    """

    def set_social_interactions(self, p_opp=0, *args, **kwargs):
        from opdynamics.dynamics.socialinteractions import (
            compute_connection_probabilities_opp,
        )

        conn_method = kwargs.pop("conn_method", compute_connection_probabilities_opp)
        super().set_social_interactions(
            *args,
            conn_method=conn_method,
            p_opp=p_opp,
            **kwargs,
        )


class NoisySocialNetwork(SocialNetwork):
    """Parent class for adding a noise term ``D`` to the dynamics of a network."""

    def __init__(self, *args, name="noisy SocialNetwork", **kwargs):
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


class OpenChamber(NoisySocialNetwork):
    """Network with *external* noise."""

    # noinspection PyTypeChecker
    def __init__(self, *args, name="open SocialNetwork", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.diffusion: diffeq = None
        self.diff_args = ()

    def set_dynamics(self, *args, D=0.01, **kwargs):
        """Network with *external* noise.

        External noise:
        -----

        .. math::
            \\dot{x}_i = K \\cdot \\sum_{j=1} A_{ij}(t) \\cdot \\tanh{(\\alpha \\cdot x_j)} + D \\cdot \\xi (t)

        Where :math:`\\xi (t)` is the Wiener Process.

        :param D: Strength of noise.

        """
        from opdynamics.dynamics.opinions import partial_diffusion

        # assign drift as before, aka dy_dt
        super().set_dynamics(D=D)

        # assign diffusion term
        self.diffusion = partial(partial_diffusion, self.rn, D, self.N)

    def run_network(
        self,
        dt: float = 0.01,
        t_dur: float = 0.05,
        method: str = "Euler-Maruyama",
    ):
        """Dynamics are no longer of an ordinary differential equation so we can't use scipy.solve_ivp anymore"""

        from opdynamics.integrate.solvers import SDE_INTEGRATORS, solve_sde

        t_span = self._setup_run(dt, t_dur)
        args = (*self._args(), dt)
        diff_args = (dt, *self.diff_args)
        if method in SDE_INTEGRATORS:
            # use a custom method in `opdynamics.utils.integrators`
            self.result: SolverResult = solve_sde(
                self.dy_dt,
                self.diffusion,
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


class ContrastChamber(NoisySocialNetwork):
    """Network with noise by contrasting the opinions of a select agent to every other agent."""

    # noinspection PyTypeChecker
    def __init__(self, *args, name="contrast SocialNetwork", **kwargs):
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
        super().set_dynamics(D=D)
        super_dy_dt = self.dy_dt

        self._idx = np.round(self.rn.uniform(0, self.N - 1, size=self.N), 0).astype(int)
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


class SampleChamber(NoisySocialNetwork):
    """
    Provide a mean sample of opinions to each agent.

    see https://en.wikipedia.org/wiki/Central_limit_theorem
    """

    # noinspection PyTypeChecker
    def __init__(self, *args, name="sample chamber", **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self._sample_size: int = 0
        self._num_samples: int = None
        self._sample_means: float = 0.0
        self._sample_method: Union[str, Callable] = None
        self._background: bool = True

    def set_dynamics(
        self,
        D: float = 0,
        sample_size: int = 20,
        sample_method: str = "full",
        num_samples: int = None,
        background: bool = True,
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
        self._num_samples = num_samples if num_samples is not None else self.N

        if not (self._num_samples == 1 or self._num_samples == self.N):
            raise ValueError(
                "num_samples must be either None (which becomes N) or 1 to be broadcast properly in the dynamics"
            )

        if background:
            self.dy_dt = sample_dy_dt
        else:
            self.dy_dt = sample_dy_dt_activity

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
            *args, self._sample_method[1], self, self._sample_size, self._num_samples
        )

    def __repr__(self):
        return (
            f"{super().__repr__()}"
            f" sample_method={self._sample_method[0]}"
            f" n={self._sample_size}"
            f" num_samples={self._num_samples}"
            f" background={self._background}"
        )


def example():
    """Simple example to show how to run a simulation and display some results."""
    from opdynamics.visualise.vissocialnetwork import VisSocialNetwork

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
    t_dur = 0.5

    sn = SampleChamber(num_agents, m, K, alpha, seed=1337)
    vis = VisSocialNetwork(sn)

    sn.set_activities(activity_distribution, gamma, epsilon, 1)
    vis.show_activities()
    vis.show_activity_vs_opinion()

    sn.set_connection_probabilities(beta=beta)
    sn.set_social_interactions(r=r, dt=dt, t_dur=t_dur)
    sn.set_dynamics()

    sn.run_network(dt=dt, t_dur=t_dur)
    vis.show_opinions(color_code=False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    example()

    plt.show()
