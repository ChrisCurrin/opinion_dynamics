from functools import lru_cache
import logging
import numpy as np
from tqdm import tqdm

from opdynamics.networks.echochamber import EchoChamber

logger = logging.getLogger("social interaction")


class SocialInteraction(object):
    """
    Compute the social interactions for the associated EchoChamber at time of request by calling
    ``si_object[<index>]``.

    The matrix for ``<index>`` is cached using ``functools.lru_cache`` so repeated calls for the same index are efficient.

    The use of calling a SocialInteraction object with an index ensures the dt time scale for social interactions is
    maintained even if the actual simulation is computed at a different dt, depending on method (such as RK45).

    Arguments:
    ------
    * ``ec`` - EchoChamber to generate the social interaction from.
    * ``p_mutual_interaction`` - 'r', the probability of a mutual interaction between agents i and j.

    Methods:
    ------
    * ``accumulate(t_idx)`` - number of interactions up to ``t_idx``. If ``t_idx`` is provided, instance must have
        ``_time_mat``.

    Properties:
    ---------
    * ``accumulator`` - keep track of the total number of interactions between agents.

    Private Properties
    ---------
    * ``_time_mat`` - adjacency matrix at each time point. Used to store eagerly computed results.
    * ``_last_adj_mat`` - keep a reference to the newest adjacency matrix of social interactions. Retrieved via [-1]

    """

    def __init__(self, ec: EchoChamber, p_mutual_interaction: float):
        self.ec = ec
        self.p_mutual_interaction = p_mutual_interaction
        self._accumulator = np.zeros((ec.N, ec.N), dtype=int)
        self._last_adj_mat = None
        self._time_mat = None

        assert (
            0 <= p_mutual_interaction <= 1
        ), "p_mutual_interaction is a probability between 0 and 1"
        logger.debug(
            f"Social Interaction for {ec.name} initialised with r={p_mutual_interaction}."
        )

    def eager(self, t_end: float, dt: float):
        """Pre-compute social interactions (the adjacency matrix) for each time step until t_end."""
        logger.info(f"eagerly computing {1 + int(t_end/dt)} adjacency matrices...")
        t_arr = np.arange(0, t_end + dt, dt)
        self._time_mat = np.zeros((len(t_arr), self.ec.N, self.ec.N), dtype=int)
        active_thresholds = self.ec.rn.random(size=len(t_arr))
        for t_idx, t_point in tqdm(enumerate(t_arr)):
            self._time_mat[t_idx, :, :] = get_social_interaction(
                self.ec, active_thresholds[t_idx], self.p_mutual_interaction
            )
        logger.debug(f"adjacency matrix has shape = {self._time_mat.shape}")

    def accumulate(self, t_idx=None):
        """The total number of interactions between agents i and j (matrix)"""
        if self._time_mat is not None:
            if t_idx is None:
                return np.sum(self._time_mat, axis=0)
            return np.sum(self._time_mat[:t_idx], axis=0)
        elif t_idx is not None and t_idx != -1:
            raise IndexError(
                f"Accumulate called with t_idx={t_idx} but adj_mat is computed lazily. "
                f"Call ``eager(<dt>, <t_end>)`` first or set `lazy=False`."
            )
        return self._accumulator

    # create a property for t_idx=None
    accumulator = property(accumulate)

    @lru_cache(maxsize=128)
    def __getitem__(self, item: int):
        if self._time_mat is not None:
            return self._time_mat[item]

        if item == -1:
            return self._last_adj_mat

        self._last_adj_mat = get_social_interaction(
            self.ec, self.ec.rn.random(), self.p_mutual_interaction
        )
        self._accumulator += self._last_adj_mat
        return self._last_adj_mat

    def __repr__(self):
        return f"Aij[r={self.p_mutual_interaction}]"


# TODO: Test
def get_connection_probabilities(ec: EchoChamber, beta: float = 0.0):
    """For agent `i`, the probability of connecting to agent `j` is a function of the absolute strength of
    their opinions and a beta param, relative to all of the differences between an agent i and every other agent.

    .. math::
        p_{ij} = \\frac{|x_i - x_j|^{-\\beta}}{\sum_j |x_i - x_j|^{-\\beta}}

    :param ec: Echo chamber object so we know
        1) number of agents
        2) agent opinions
    :param beta: Power law decay of connection probability. Decay when beta>0, increase when beta<0.
        When beta=0, then connection probabilities are uniform.

    """
    # create N * N matrix of opinions
    mat_opinions = np.tile(ec.opinions, ec.N)
    # compute magnitude (between agent i and every other agent)*N agents
    mag = np.abs(ec.opinions - mat_opinions)
    self_mask = np.identity(ec.N)
    mag[self_mask] = 0
    p_conn = np.power(mag, -beta)
    p_conn /= np.sum(p_conn, axis=1)
    return p_conn


def get_social_interaction(
    ec: EchoChamber, active_threshold: float, p_mutual_interaction: float
):
    """
    Compute the social interactions to occur within an EchoChamber.

    :param ec: Echo chamber object so we know
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
    adj_mat = np.zeros((ec.N, ec.N), dtype=int)
    active_agents = np.where(ec.activities >= active_threshold)[0]
    is_mutual = ec.rn.random(size=(len(active_agents), ec.m)) < p_mutual_interaction
    for loop_i, a_i in enumerate(active_agents):
        # loop_i is how far through the active_agents the loop is
        # a_i is the index of the agent
        ind: np.ndarray = ec.rn.choice(
            ec.N,  # choose indices for
            size=ec.m,  # other distinct agents
            replace=False,  # (which must be unique)
            p=ec.p_conn[a_i],  # with these probabilities
        )
        # agent i influences agents j (i -> j, Aji = 1)
        adj_mat[ind, a_i] = 1
        # reciprocal interaction (agent i is influenced by agents j, Aij = 1), given `p_mutual_interaction`
        # keep an interaction if it exists already, or add a new mutual interaction
        adj_mat[a_i, ind] = np.logical_or(adj_mat[a_i, ind], is_mutual[loop_i])
    return adj_mat


def get_social_interaction_exp(
    ec: EchoChamber, active_threshold: float, p_mutual_interaction: float
):
    """
        Compute the social interactions to occur within an EchoChamber.

        ** Experimental vectorised version **

        https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops

        :param ec: Echo chamber object so we know
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
    adj_mat = np.zeros((ec.N, ec.N), dtype=int)
    active_agents = np.where(ec.activities >= active_threshold)[0]
    is_mutual = ec.rn.random(size=(len(active_agents), ec.m)) < p_mutual_interaction
    p = ec.p_conn[active_agents].ravel() / np.sum(ec.p_conn[active_agents])
    a = np.tile(np.arange(ec.N), len(active_agents)).ravel()

    ind: np.ndarray = ec.rn.choice(
        a,  # choose indices for
        size=(len(active_agents), ec.m),  # other distinct agents
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
