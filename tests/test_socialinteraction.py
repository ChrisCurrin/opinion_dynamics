from typing import Optional
from unittest import TestCase

import numpy as np

from opdynamics.dynamics.socialinteractions import (
    compute_connection_probabilities,
    compute_connection_probabilities_v1,
    compute_connection_probabilities_v2,
    compute_social_interaction,
    get_social_interaction_exp,
)
from opdynamics.socialnetworks import SocialNetwork


class TestSocialInteraction(TestCase):
    def setUp(self) -> None:
        self.sn = SocialNetwork(1000, m=10, K=3.0, alpha=3.0)

    def test_get_social_interaction(self):
        from scipy.stats import norm

        # required setup
        # set activities to all be 1 (all agents interact at every time step)
        self.sn.set_activities(norm, 1, 0)
        self.sn.set_social_interactions(beta=0)

        # no mutual interactions
        r = 0
        # all active
        active_threshold = 0

        self.sn.seed = 42
        adj_mat: np.ndarray = compute_social_interaction(self.sn, active_threshold, r)
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


class TestComputeConnectionProbabilities(TestCase):
    def setUp(self) -> None:
        # opinions shape is (n_topics, n_agents)
        self.opinions = np.expand_dims(
            np.array([-0.6, -0.5, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), 0
        )

        self.opinions_n_topics = np.array(
            [
                [-0.6, -0.5, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [-0.6, -0.5, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [-0.6, -0.5, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            ]
        )

    def test_compute_connection_probabilities_beta_zero(self):
        beta: float = 0.0
        opinions = self.opinions
        N = opinions.shape[1]
        p_conn = compute_connection_probabilities(opinions, beta)
        # check that all probabilities are the same, except along the diagonal

        expected_p = 1 / (N - 1)
        expected_p_conn = np.ones((N, N)) * expected_p
        expected_p_conn[np.diag_indices(N)] = 0

        self.assertListEqual(
            p_conn.tolist(),
            expected_p_conn.tolist(),
            "expected all probabilities to be equal for beta=0, instead got"
            f" {p_conn}",
        )
        # also that all sum to 1
        self.assertTrue(
            np.all(np.sum(p_conn, axis=1) == 1),
            "expected all probabilities to sum to 1",
        )

    def test_all_versions(self):
        opinions = self.opinions

        for beta in [-1, 0, 1]:
            p_conn = compute_connection_probabilities(opinions, beta)
            p_conn_v2 = compute_connection_probabilities_v2(opinions, beta)
            p_conn_v1 = compute_connection_probabilities_v1(opinions, beta)
            self.assertListEqual(
                p_conn_v1.round(10).tolist(),
                p_conn_v2.round(10).tolist(),
                f"expected v1 and v2 to be equal for beta={beta}",
            )
            self.assertListEqual(
                p_conn.round(10).tolist(),
                p_conn_v2.round(10).tolist(),
                f"expected v2 and v3 to be equal for beta={beta}",
            )

    def test_multiple_topics_orthogonal(self):
        beta: float = 0.0
        opinions = self.opinions_n_topics

        # angle is a matrix between all topics
        angle = np.zeros((opinions.shape[0], opinions.shape[0]))

        p_conn = compute_connection_probabilities(opinions, beta, angle=angle)

        # check that all probabilities are the same, except along the diagonal
        expected_p = 1 / (len(opinions) - 1)
        expected_p_conn = np.ones((len(opinions), len(opinions))) * expected_p
        expected_p_conn[np.diag_indices(len(opinions))] = 0

        self.assertListEqual(
            p_conn.tolist(),
            expected_p_conn.tolist(),
            "expected all probabilities to be equal for beta=0, instead got"
            f" {p_conn}",
        )
