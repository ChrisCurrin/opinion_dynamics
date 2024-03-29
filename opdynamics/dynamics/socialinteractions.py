"""
# Activity-driven (AD) dynamics

------

## Methods
====

* ``get_connection_probabilities``
    get connection probabilities from opinions and a beta factor
    * ``get_connection_probabilities_opp``
        same as above, but with an additional term ``p_opp`` making it more likely to connect
        with an agent holding an opposing opinion and less likely to connect with an agent holding a similar
        opinion.
    * ``get_connection_probabilities_exp``
        same as :meth:``get_connection_probabilities`` but optimised (experimental)
* ``get_social_interaction``
    get social interaction matrix from an activity threshold, reciprocity factor, and connection probability matrix
    * ``get_social_interaction_exp``
        same as :meth:``get_social_interaction`` but optimised (experimental)

## Classes
====
- ``SocialInteraction``
    Interface to calculate connection probabilities and social interaction matrix at discrete time steps

"""
import atexit
import inspect
import json
import logging
import os
from functools import lru_cache
from typing import Tuple, Union

import numpy as np
from opdynamics.socialnetworks import SocialNetwork
from opdynamics.utils.cache import NpEncoder, get_hash_filename
from opdynamics.utils.decorators import hashable

logger = logging.getLogger("social interaction")


def get_connection_probabilities(sn: SocialNetwork, beta: float = 0.0, **conn_kwargs):
    """For agent `i`, the probability of connecting to agent `j` is a function of the absolute strength of
    their opinions and a beta param, relative to all of the differences between an agent i and every other agent.

    .. math::
        p_{ij} = \\frac{|x_i - x_j|^{-\\beta}}{\sum_j |x_i - x_j|^{-\\beta}}

    :param sn: SocialNetwork object so we know
        1) number of agents
        2) agent opinions
    :param beta: Power law decay of connection probability. Decay when beta>0, increase when beta<0.
        When beta=0, then connection probabilities are uniform.

    """
    # create N * N matrix of opinions
    p_conn = np.zeros(shape=(sn.N, sn.N))
    for i in range(sn.N):
        # compute magnitude (between agent i and every other agent)*N agents
        mag = np.abs(sn._opinions[i] - sn._opinions)
        mag[i] = np.nan
        p_conn[i] = np.power(mag, -beta)
        p_conn[i, i] = 0
        p_conn[i] /= np.sum(p_conn[i])
    return p_conn


def get_connection_probabilities_opp(
    sn: SocialNetwork, beta: float = 0.0, p_opp: float = 0.0, **conn_kwargs
):
    """For agent `i`, the probability of connecting to agent `j` is a function of the absolute strength of
    their opinions and a beta param, relative to all of the differences between an agent i and every
    other agent.

    .. math::
        p_{ij} = \\frac{|x_i - x_j|^{-\\beta}}{\\sum_j |x_i - x_j|^{-\\beta}}

    :param beta: Power law decay of connection probability. Decay when beta>0, increase when beta<0.
        When beta=0, then connection probabilities are uniform.
    :param p_opp: Probability of inverting the power-law from decay to gain, making it more likely to connect
        with an agent holding an opposing opinion and less likely to connect with an agent holding a similar
        opinion.
    """
    if p_opp > 0:
        betas = sn.rn.choice([beta, -beta], size=sn.N, p=[p_opp, 1 - p_opp])
    else:
        betas = beta
    return get_connection_probabilities(sn, beta=betas, **conn_kwargs)


# TODO: Test
def get_connection_probabilities_exp(
    sn: SocialNetwork, beta: float = 0.0, **conn_kwargs
):
    """For agent `i`, the probability of connecting to agent `j` is a function of the absolute strength of
    their opinions and a beta param, relative to all of the differences between an agent i and every other agent.

    .. math::
        p_{ij} = \\frac{|x_i - x_j|^{-\\beta}}{\sum_j |x_i - x_j|^{-\\beta}}

    :param sn: SocialNetwork object so we know
        1) number of agents
        2) agent opinions
    :param beta: Power law decay of connection probability. Decay when beta>0, increase when beta<0.
        When beta=0, then connection probabilities are uniform.

    """
    # create N * N matrix of opinions
    mat_opinions = np.tile(sn._opinions, sn.N)
    # compute magnitude (between agent i and every other agent)*N agents
    mag = np.abs(sn._opinions - mat_opinions)
    self_mask = np.identity(sn.N)
    mag[self_mask] = 0
    p_conn = np.power(mag, -beta)
    p_conn /= np.sum(p_conn, axis=1)
    return p_conn


def get_social_interaction(
    sn: SocialNetwork,
    active_threshold: float,
    p_mutual_interaction: float,
    p_conn: np.ndarray,
):
    """
    Compute the social interactions to occur within an SocialNetwork.

    :param sn: SocialNetwork object so we know
        1) number of agents
        2) number of other agents to interact with
        3) the probability of interacting with each other agent
    :param active_threshold: Threshold for an agent to be active.
    :param p_mutual_interaction: Probability that an interaction is mutual (matrix becomes symmetrical if all
        interactions are mutual).
        If 1: create symmetric matrix by not distinguishing between i->j  and j->i
        If 0: a non-symmetric matrix means an agent ignores external interactions
    :param p_conn: connection probabilities to use
    :return: Adjacency matrix for interactions between agents.
    :rtype: np.ndarray
    """
    adj_mat = np.zeros((sn.N, sn.N), dtype=int)
    active_agents = np.where(sn.activities >= active_threshold)[0]
    is_mutual = sn.rn.random(size=(len(active_agents), sn.m)) < p_mutual_interaction
    for loop_i, a_i in enumerate(active_agents):
        # loop_i is how far through the active_agents the loop is
        # a_i is the index of the agent
        ind: np.ndarray = sn.rn.choice(
            sn.N,  # choose indices [0, N-1] for
            size=sn.m,  # m other distinct agents
            replace=False,  # (which must be unique)
            p=p_conn[a_i],  # with these probabilities
        )
        # agent i influences agents j (i -> j, Aji = 1)
        adj_mat[ind, a_i] = 1
        # reciprocal interaction (agent i is influenced by agents j, Aij = 1), given `p_mutual_interaction`
        # keep an interaction if it exists already, or add a new mutual interaction
        adj_mat[a_i, ind] = np.logical_or(adj_mat[a_i, ind], is_mutual[loop_i])
    return adj_mat


def get_social_interaction_exp(
    sn: SocialNetwork, active_threshold: float, p_mutual_interaction: float
):
    """
    Compute the social interactions to occur within an SocialNetwork.

    ** Experimental vectorised version **

    https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops

    :param sn: SocialNetwork object so we know
        1) number of agents
        2) number of other agents to interact with
        3) the probability of interacting with each other agent
    :param active_threshold: Threshold for an agent to be active.
    :param p_mutual_interaction: Probability that an interaction is mutual (matrix becomes symmetrical if all
        interactions are mutual).
        If 1: create symmetric matrix by not distinguishing between i->j  and j->i
        If 0: a non-symmetric matrix means an agent ignores external interactions
    :return: Adjacency matrix for interactions between agents.
    :rtype: np.ndarray
    """
    adj_mat = np.zeros((sn.N, sn.N), dtype=int)
    active_agents = np.where(sn.activities >= active_threshold)[0]
    is_mutual = sn.rn.random(size=(len(active_agents), sn.m)) < p_mutual_interaction
    p = sn.p_conn[active_agents].ravel() / np.sum(sn.p_conn[active_agents])
    a = np.tile(np.arange(sn.N), len(active_agents)).ravel()

    ind: np.ndarray = sn.rn.choice(
        a,  # choose indices for
        size=(len(active_agents), sn.m),  # other distinct agents
        replace=False,  # (which must be unique)
        p=p,  # with these probabilities
        shuffle=False,
    )

    # agent i influences agents j (i -> j, Aji = 1)
    adj_mat[ind, np.arange(adj_mat.shape[0])[:, None]] = 1
    # reciprocal interaction (agent i is influenced by agents j, Aij = 1), given `p_mutual_interaction`
    # keep an interaction if it exists already, or add a new mutual interaction
    adj_mat[np.arange(adj_mat.shape[0])[:, None], ind] = np.logical_or(
        adj_mat[np.arange(adj_mat.shape[0])[:, None], ind], is_mutual
    )

    return adj_mat


@hashable
class SocialInteraction(object):
    """
    Compute the social interactions for the associated SocialNetwork at time of request by calling
    ``si_object[<index>]``.

    The matrix for ``<index>`` is cached using ``functools.lru_cache`` so repeated calls for the same index are efficient.

    The use of calling a SocialInteraction object with an index ensures the dt time scale for social interactions is
    maintained even if the actual simulation is computed at a different dt, depending on method (such as RK45).

    Arguments:
    ------
    * ``sn`` - SocialNetwork to generate the social interaction from.
    * ``p_mutual_interaction`` - 'r', the probability of a mutual interaction between agents i and j.
    * ``conn_method`` - the method used to calculate connection probabilities
    * ``update_conn`` - whether to update connection probabilities at every call (every `dt`).
    * ``conn_kwargs`` - Keyword arguments for ``conn_method``. E.g. ``beta``.

    Methods:
    ------
    * ``accumulate(t_idx)`` - number of interactions up to ``t_idx``. If ``t_idx`` is provided, instance must have
        ``_time_mat``.

    Properties:
    ---------
    * ``accumulator`` - keep track of the total number of interactions between agents.
    * ``eager`` - compute all matrices up to ``T`` in steps of ``dt``.

    Private Properties
    ---------
    * ``_time_mat`` - adjacency matrix at each time point. Used to store eagerly computed results.
    * ``_last_adj_mat`` - keep a reference to the newest adjacency matrix of social interactions. Retrieved via [-1]
    * ``_p_conn`` - keep a reference to the latest connection probabilities matrix.

    """

    def __init__(
        self,
        sn: SocialNetwork,
        p_mutual_interaction: float,
        conn_method=get_connection_probabilities,
        update_conn=False,
        **conn_kwargs,
    ):
        self.sn = sn
        self.p_mutual_interaction = p_mutual_interaction
        self.conn_method = conn_method
        # get (keyword) arguments for the connection probability method being used
        required_conn_kwargs = set(inspect.getfullargspec(conn_method)[0])
        if "sn" in required_conn_kwargs:
            required_conn_kwargs.remove("sn")
        # only save required (keyword) arguments
        conn_kwargs = {
            k: v for k, v in conn_kwargs.items() if k in required_conn_kwargs
        }
        self.conn_kwargs = conn_kwargs
        self._p_conn: np.ndarray = (
            self.conn_method(sn, **conn_kwargs) if not update_conn else None
        )
        self._update_conn = update_conn
        self._accumulator = np.zeros((sn.N, sn.N), dtype=int)
        self._last_adj_mat: np.ndarray = None
        self._time_mat: np.memmap = None

        assert (
            0 <= p_mutual_interaction <= 1
        ), "p_mutual_interaction is a probability between 0 and 1"
        logger.debug(f"Social Interaction for {sn.name} initialised {self}.")

    def compress(self, overwrite=True):
        if self._time_mat is not None:
            adj_mat_file_compressed = self._time_mat.filename.replace(".dat", ".npz")
            if overwrite or not os.path.exists(adj_mat_file_compressed):
                logger.debug(f"saving full adj_mat to '{adj_mat_file_compressed}'")
                # save compressed version
                np.savez_compressed(adj_mat_file_compressed, time_mat=self._time_mat)
                new_time_mat = np.load(adj_mat_file_compressed, mmap_mode="r+")["time_mat"]
                fname = self._time_mat.filename
                # delete previous mmap file to explicitly clear storage
                self.clear()
                # link to stored version
                self._time_mat = new_time_mat
                logger.debug(f"...saved full adj_mat and deleted memory map")

    def clear(self):
        del self._time_mat
        self._time_mat = None

    def store_interactions(self, dt: float, t_dur: float):
        """Initialise the object to store social interactions (the adjacency matrix) for each time step until t_dur."""
        logger.info(f"storing {1 + int(t_dur/dt)} adjacency matrices...")
        t_arr = np.arange(0, t_dur + dt, dt)
        adj_mat_memmap_file = get_hash_filename(self, "dat", extra=f"{self.sn._seed}")

        is_extend_time = self._time_mat is not None

        self._time_mat = np.memmap(
            adj_mat_memmap_file,
            dtype=int,
            mode="r+" if os.path.exists(adj_mat_memmap_file) else "w+",
            shape=(len(t_arr), self.sn.N, self.sn.N),
        )

        # set up hook to clean up upon system exit
        if not is_extend_time:
            atexit.register(self.clear)

        logger.debug(f"adjacency matrix has shape = {self._time_mat.shape}")

    def accumulate(self, t_idx: Union[int, slice, Tuple[int, int]] = -1):
        """The total number of interactions between agents i and j (matrix).

        If `t_idx` is provided, either the accumulation up to that index (if an int) or the accumulation
        between 2 points (slice like 5:10) is returned.
        Generally, `t_idx` will only work as expected if `store_interactions` was set to True.
        The exception is `t_idx=-1`, the default value, which returns the total number of interactions regardless of the value of
        `store_interactions`.
        """
        if isinstance(t_idx, int) and t_idx == -1:
            return self._accumulator
        elif self._time_mat is not None:
            if isinstance(t_idx, int):
                return np.sum(self._time_mat[:t_idx], axis=0)
            elif np.iterable(t_idx):
                return np.sum(self._time_mat[t_idx[0] : t_idx[1]], axis=0)
            # slice
            return np.sum(self._time_mat[t_idx], axis=0)
        else:
            raise IndexError(
                f"Accumulate called with t_idx={t_idx} but adj_mat is not stored. "
                f"Call ``store_interactions(<dt>, <t_dur>)`` first or set `cache='all'` for simulations."
            )

    # Cumulative adjacency matrix create a property for t_idx=-1
    total = property(accumulate)
    # backwards compatibility
    accumulator = total

    @lru_cache(maxsize=128)
    def _getitem__specific(self, item: int):
        if item == -1:
            return self._last_adj_mat

        if self._time_mat is not None and np.sum(self._time_mat[item]) > 0:
            # we are storing interactions and have already computed this item (not every value is 0)
            return self._time_mat[item]

        # compute interactions

        if self._update_conn:
            self._p_conn = self.conn_method(self.sn, **self.conn_kwargs)

        self._last_adj_mat = get_social_interaction(
            self.sn, self.sn.rn.random(), self.p_mutual_interaction, self._p_conn
        )
        # update accumulator
        self._accumulator += self._last_adj_mat

        if self._time_mat is not None:
            # update full matrix (if applicable)
            self._time_mat[item, :, :] = self._last_adj_mat

        return self._last_adj_mat

    def _getitem__slice(self, key: slice):
        return self._time_mat[key]

    def __getitem__(self, item: Union[int, slice]):
        if isinstance(item, slice):
            return self._getitem__slice(item)
        return self._getitem__specific(item)

    def __repr__(self):
        return (
            f"Aij[r={self.p_mutual_interaction}] "
            f"{self.conn_method.__name__}("
            f"{json.dumps(self.conn_kwargs, sort_keys=True, separators=(',', ':'), cls=NpEncoder)})"
            f"{'(t)' if self._update_conn else ''}"
        )
