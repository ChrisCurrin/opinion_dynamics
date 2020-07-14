import numpy as np

from unittest import TestCase

from opdynamics.networks.echochamber import EchoChamber
from opdynamics.networks.socialinteraction import (
    get_social_interaction,
    get_social_interaction_exp,
)


class TestSocialInteraction(TestCase):
    def setUp(self) -> None:

        self.ec = EchoChamber(1000, m=10, K=3.0, alpha=3.0)

    def test_get_social_interaction(self):
        from scipy.stats import norm

        # required setup
        # set activities to all be 1 (all agents interact at every time step)
        self.ec.set_activities(norm, 1, 0)
        self.ec.set_connection_probabilities(beta=0)

        # no mutual interactions
        r = 0
        # all active
        active_threshold = 0

        self.ec.rn = np.random.default_rng(42)
        adj_mat = get_social_interaction(self.ec, active_threshold, r)
        self.assertTrue(np.all(np.sum(adj_mat, axis=0) == self.ec.m))
        self.assertTrue(
            np.all(adj_mat.diagonal() == 0), "expected no interactions with self"
        )
        self.ec.rn = np.random.default_rng(self.ec.rn)
        adj_mat_exp = get_social_interaction_exp(self.ec, active_threshold, r)
        self.assertTrue(
            np.all(np.sum(adj_mat_exp, axis=0) == self.ec.m),
            "every agent must interact with 10 other agents (normal distribution, no threshold)",
        )
        self.assertTrue(
            np.all(adj_mat_exp.diagonal() == 0), "expected no interactions with self"
        )
