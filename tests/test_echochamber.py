import logging
import inspect
import numpy as np

from unittest import TestCase

from opdynamics.dynamics.echochamber import EchoChamber

# noinspection PyUnusedName
from opdynamics.integrate.types import OdeResult

logger = logging.getLogger("test_echochamber")


class TestEchoChamber(TestCase):
    def setUp(self) -> None:
        self.ec = EchoChamber(1000, m=10, K=3.0, alpha=3.0)

    def test_init_opinions(self):
        self.ec.opinions = np.zeros(shape=self.ec.N)
        self.assertTrue(np.all(self.ec.opinions == 0))
        min_val = -1
        max_val = 1
        self.ec.init_opinions(min_val, max_val)
        self.assertGreaterEqual(np.min(self.ec.opinions), min_val)
        self.assertLessEqual(np.max(self.ec.opinions), max_val)
        self.assertTrue(
            self.ec.opinions.shape[0] == self.ec.N,
            "each agent in the echo chamber must have an opinion",
        )

    def test_set_activities(self):
        from scipy.stats import powerlaw
        from opdynamics.utils.distributions import negpowerlaw

        gamma = 2
        min_val = 0.1
        max_val = 0.5
        self.ec.set_activities(negpowerlaw, gamma, min_val, max_val, dim=1)
        self.assertGreaterEqual(np.min(self.ec.activities), min_val)
        self.assertLessEqual(np.max(self.ec.activities), max_val)
        self.assertTrue(
            self.ec.activities.shape[0] == self.ec.N,
            "each agent in the echo chamber must have an activity probability",
        )
        self.ec.set_activities(powerlaw, gamma, min_val, max_val, dim=1)
        self.assertGreaterEqual(np.min(self.ec.activities), min_val)
        self.assertLessEqual(np.max(self.ec.activities), max_val)
        self.assertTrue(
            self.ec.activities.shape[0] == self.ec.N,
            "each agent in the echo chamber must have an activity probability",
        )

        # TODO: test other distributions
        # mu = 1
        # sigma = 2
        # self.ec.N = 100000  # just for distribution purpose
        # self.ec.set_activities(norm, mu, sigma, dim=1, inverse=False)
        # self.assertAlmostEquals(np.mean(self.ec.activities), mu)
        # self.assertAlmostEquals(np.std(self.ec.activities), sigma)

    def test_set_connection_probabilities(self):

        # probabilities must sum to 1 (by definition)
        for beta in np.arange(-2, 2, 0.5):
            self.ec.set_connection_probabilities(beta=beta)
            p_conn_i = np.round(np.sum(self.ec.p_conn, axis=1), 4)
            self.assertTrue(
                np.all(p_conn_i == 1),
                msg=f"probabilities do not sum to 1 for beta={beta}. p_conn_i={p_conn_i}",
            )
            self.assertAlmostEqual(
                np.sum(self.ec.p_conn),
                self.ec.N,
                places=4,
                msg=f"there should be {self.ec.N} probability distributions for beta={beta}",
            )
            diag_v = np.diag(self.ec.p_conn)
            self.assertTrue(np.all(diag_v == 0), "self-connections must be 0")

        # beta = 0 is the same as a uniform distribution
        self.ec.set_connection_probabilities(beta=0)
        diag_mat = np.diagflat([1] * self.ec.N).astype(bool)
        self.assertTrue(
            np.all(np.round(self.ec.p_conn[~diag_mat], 4) == (1 / self.ec.N)),
            "expected uniform distribution for beta=0",
        )

    def test_set_social_interactions(self):
        from opdynamics.dynamics.socialinteraction import SocialInteraction
        from scipy.stats import norm

        # this must be first (before setting activities and connection probabilities)
        with self.assertRaises(RuntimeError):
            self.ec.set_social_interactions(0, lazy=True)

        # required setup
        # set activities to all be 1 (all agents interact at every time step)
        self.ec.set_activities(norm, 1, 0)
        self.ec.set_connection_probabilities(beta=0)

        for lazy in [True, False]:
            self.ec.set_social_interactions(0, lazy=lazy, t_end=0.5, dt=0.01)
            self.assertTrue(isinstance(self.ec.adj_mat, SocialInteraction))
            # get a matrix at time 0
            mat = self.ec.adj_mat[0]
            self.assertTrue(
                isinstance(mat, np.ndarray),
                "adjacency matrix expected to be a numpy array",
            )
            self.assertEqual(
                mat.ndim, 2, "adjacency matrix expected to be a 2D numpy array"
            )
            # access again and check that it's the same object
            mat2 = self.ec.adj_mat[0]

            self.assertTrue(
                np.all(mat == mat2),
                "expected to use the cache to retrieve the same matrix",
            )

        # check for different r's
        for r in [0, 0.2, 0.5, 0.8, 1]:
            self.ec.set_social_interactions(r, lazy=True)
            mat = self.ec.adj_mat[0]
            total = np.sum(mat)

            self.assertGreaterEqual(
                total,
                self.ec.N * self.ec.m * (1 + r * 0.95),
                f"expected every agent to interact with at least {self.ec.m} other agents. This means that for "
                f"r={r}, there should be at least {self.ec.N*self.ec.m*(1 + r*0.95)} interactions (within 95% "
                f"to account for some overlap)",
            )
            self.assertLessEqual(
                total,
                self.ec.N * self.ec.m * (1 + r),
                f"expected no more interactions than governed by mutual interactions (r={r})",
            )
            if r == 0:
                self.assertEqual(
                    total,
                    self.ec.N * self.ec.m,
                    "when r=0, expected every agent to interact with 10 other agents only",
                )
            if r < 1:
                self.assertFalse(
                    np.all(mat == mat.T), "when r<1, matrix should *not* be symmetrical"
                )
            self.assertTrue(
                np.all(mat.diagonal() == 0), "expected no interactions with self"
            )

        # eager mode requires t_end and dt
        with self.assertRaises(Exception):
            self.ec.set_social_interactions(0, lazy=False)

    def test_set_dynamics(self):
        self.assertIsNone(self.ec.dy_dt)
        self.ec.set_dynamics()
        self.assertIsNotNone(self.ec.dy_dt)
        self.assertTrue(
            hasattr(self.ec.dy_dt, "__call__"), "dy_dt must be callable",
        )

        func = inspect.getfullargspec(self.ec.dy_dt)
        self.assertListEqual(
            func.args, ["t", "y"], "dynamics must have (t, y, *args) as its signature"
        )
        self.assertIsNotNone(
            func.varargs, "dynamics must have '*args' in its signature"
        )
        self.assertListEqual(
            func.kwonlyargs, [], "expected no keyword arguments in dynamics"
        )
        self.assertIsNone(func.varkw, "expected no variable keywords in dynamics")
        self.assertTrue(
            inspect.signature(self.ec.dy_dt).return_annotation == np.ndarray,
            "expected numpy array return.",
        )

    def test_run_network(self):
        from opdynamics.utils.distributions import negpowerlaw
        from opdynamics.integrate.types import SolverResult

        self.ec.alpha = 3  # controversialness of issue (sigmoidal shape)
        self.ec.K = 3  # social interaction strength
        epsilon = 1e-2  # minimum activity level with another agent
        gamma = 2.1  # power law distribution param
        beta = 3  # power law decay of connection probability
        r = 0.5  # probability of a mutual interaction
        activity_distribution = negpowerlaw

        dt = 0.01
        T = 0.1
        # this must be first (before setting activities and connection probabilities)
        with self.assertRaises(RuntimeError):
            self.ec.run_network()

        # required setup
        self.ec.set_activities(negpowerlaw, gamma, epsilon)
        self.ec.set_connection_probabilities(beta=beta)
        self.ec.set_social_interactions(r=r, lazy=True)
        self.ec.set_dynamics()

        # run network using scipy solver (Rk45) and custom solver (Euler)
        min_total_iter = 0
        for method in ["RK45", "Euler"]:
            self.ec.run_network(dt=dt, t_end=T, method=method)
            min_total_iter += T // dt
            self.assertTrue(isinstance(self.ec.result, OdeResult))
            self.assertTrue(hasattr(self.ec.result, "t"))
            self.assertTrue(hasattr(self.ec.result, "y"))
            self.assertGreaterEqual(self.ec.result.t.shape[0], min_total_iter)
            self.assertGreaterEqual(self.ec.result.y.shape[1], min_total_iter)
            self.assertEqual(self.ec.result.y.shape[0], self.ec.N)

    def test_get_mean_opinion(self):
        from opdynamics.utils.distributions import negpowerlaw

        self.ec.alpha = 0.05  # controversialness of issue (sigmoidal shape)
        self.ec.K = 3  # social interaction strength
        epsilon = 1e-2  # minimum activity level with another agent
        gamma = 2.1  # power law distribution param
        beta = 2  # power law decay of connection probability
        r = 0.5  # probability of a mutual interaction

        self.ec.N = 10000
        self.ec.init_opinions(-1, 1)
        self.assertAlmostEqual(np.mean(self.ec.opinions), 0, 2)

        self.ec.set_activities(negpowerlaw, gamma, epsilon)
        self.ec.set_connection_probabilities(beta=beta)
        self.ec.set_social_interactions(r=r, lazy=True)
        self.ec.set_dynamics()

        T = 0.02
        dt = 0.01
        # run network for 2 time steps
        self.ec.run_network(dt=dt, t_end=T, method="Euler")

        t, opinion = self.ec.get_mean_opinion(0)
        self.assertAlmostEqual(opinion, 0, 2)
        t, opinion = self.ec.get_mean_opinion(dt)
        self.assertAlmostEqual(opinion, 0, 2)
        t, opinion = self.ec.get_mean_opinion(T)
        t_last, opinion_last = self.ec.get_mean_opinion(-1)
        self.assertAlmostEqual(opinion, 0, 2)
        self.assertEqual(opinion, opinion_last)

    def test_get_nearest_neighbours(self):
        self.fail()

    def test_run_params(self):
        self.fail()
