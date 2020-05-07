from functools import lru_cache

import numpy as np

from opdynamics.echochamber import EchoChamber


class SocialInteraction(object):
    """
    Compute the social interactions for the associated EchoChamber at time of request by calling
    `si_object[<index>]`.

    The matrix for `<index>` is cached using `functools.lru_cache` so repeated calls for the same index are efficient.

    The use of calling a SocialInteraction object with an index ensures the dt time scale for social interactions is
    maintained even if the actual simulation is computed at a different dt, depending on method (such as RK45).

    """

    def __init__(self, ec: EchoChamber, p_mutual_interaction: float):
        self.ec = ec
        self.p_mutual_interaction = p_mutual_interaction

        assert (
            0 <= p_mutual_interaction <= 1
        ), "p_mutual_interaction is a probability between 0 and 1"

    @lru_cache(maxsize=128)
    def __getitem__(self, item):
        get_social_interaction(self.ec, self.ec.rn.random(), self.p_mutual_interaction)


def get_social_interaction(
    ec: EchoChamber, active_threshold: float, p_mutual_interaction: float
):
    """
    Compute the social interactions to occur within an EchoChamber.

    :param ec: Echo chamber object so we know
        1) number of agents
        2) number of other agents to interact with
        3) the probability of interacting with each other agent (choose from 2).
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
    is_mutual = np.random.random(size=len(active_agents)) <= p_mutual_interaction
    for loop_i, a_i in enumerate(active_agents):
        # loop_i is how far through the active_agents the loop is
        # a_i is the index of the agent
        ind = ec.rn.choice(
            ec.N,  # choose indices for
            size=ec.m,  # other distinct agents
            replace=False,  # (which must be unique)
            p=ec.p_conn[a_i],  # with these probabilities
        )
        adj_mat[a_i, ind] = 1
        # reciprocal interactions
        adj_mat[ind, a_i] = is_mutual[loop_i]
    return adj_mat
