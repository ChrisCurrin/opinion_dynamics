import numpy as np

from unittest import TestCase

from opdynamics.socialnetworks import SocialNetwork
from opdynamics.dynamics.socialinteractions import (
    compute_social_interaction,
    get_social_interaction_exp,
)


class TestSocialInteraction(TestCase):
    def setUp(self) -> None:

        self.sn = SocialNetwork(1000, m=10, K=3.0, alpha=3.0)

    def test_get_social_interaction(self):
        from scipy.stats import norm

        # required setup
        # set activities to all be 1 (all agents interact at every time step)
        self.sn.set_activities(norm, 1, 0)
        self.sn.set_connection_probabilities(beta=0)

        # no mutual interactions
        r = 0
        # all active
        active_threshold = 0

        self.sn.rn = np.random.default_rng(42)
        adj_mat = compute_social_interaction(self.sn, active_threshold, r)
        self.assertTrue(np.all(np.sum(adj_mat, axis=0) == self.sn.m))
        self.assertTrue(
            np.all(adj_mat.diagonal() == 0), "expected no interactions with self"
        )
        self.sn.rn = np.random.default_rng(self.sn.rn)
        adj_mat_exp = get_social_interaction_exp(self.sn, active_threshold, r)
        self.assertTrue(
            np.all(np.sum(adj_mat_exp, axis=0) == self.sn.m),
            "every agent must interact with 10 other agents (normal distribution, no threshold)",
        )
        self.assertTrue(
            np.all(adj_mat_exp.diagonal() == 0), "expected no interactions with self"
        )
