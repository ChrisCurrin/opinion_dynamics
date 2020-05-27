import copy
import logging
import inspect
import os

import numpy as np

from unittest import TestCase

from opdynamics.dynamics.echochamber import EchoChamber, NoisyEchoChamber

# noinspection PyUnusedName
from opdynamics.integrate.types import OdeResult

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_echochamber")


def _predictable_interaction(ec, lazy=True):
    """Set up consistent interactions by manipulating probabilities.
    Activities: Set every agent to 100% active (mean of 1 and SD of 0).
    Connection probability: Equal probability (beta=0 yields p=1/m).
    Social Interaction: No mutual interactions (r=0).
    """
    from scipy.stats import norm

    ec.set_activities(norm, 1, 0)
    ec.set_connection_probabilities(beta=0.0)
    ec.set_social_interactions(0.0, lazy=lazy)


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
        from scipy.stats import powerlaw, norm
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

        # Normal
        mu = 1
        sigma = 0
        self.ec.N = 100
        self.ec.set_activities(norm, mu, sigma, dim=1)
        self.assertAlmostEquals(np.mean(self.ec.activities), mu)
        self.assertAlmostEquals(np.std(self.ec.activities), sigma)

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

    def test_save_load(self):
        from opdynamics.utils.distributions import negpowerlaw

        self.ec.alpha = 3  # controversialness of issue (sigmoidal shape)
        self.ec.K = 3  # social interaction strength

        dt = 0.01
        T = 0.1

        _predictable_interaction(self.ec)
        self.ec.set_dynamics()

        ec_T1 = EchoChamber(
            self.ec.N, self.ec.m, self.ec.K, self.ec.alpha, name=self.ec.name
        )
        _predictable_interaction(ec_T1)
        ec_T1.set_dynamics()

        ec_T2 = EchoChamber(
            self.ec.N, self.ec.m, self.ec.K, self.ec.alpha, name=self.ec.name
        )
        _predictable_interaction(ec_T2)
        ec_T2.set_dynamics()

        filename_T0 = self.ec._get_filename()
        self.assertFalse(os.path.exists(filename_T0))
        self.assertFalse(self.ec.load(dt, T))

        self.ec.run_network(dt, T)
        filename = self.ec._get_filename()
        try:
            saved_filename = self.ec.save()
            self.assertTrue(saved_filename, filename)
            self.assertTrue(os.path.exists(filename))

            self.assertTrue(
                ec_T1.result is None, "expected results to be None before loading"
            )
            self.assertTrue(ec_T1.load(dt, T), "did not load results as expected")
            self.assertTrue(ec_T1.result is not None, "expected results to loaded")
            self.assertFalse(ec_T2.has_results, "ec_T2 should still be a shell")

            self.assertListEqual(
                list(self.ec.result.y.ravel()),
                list(ec_T1.result.y.ravel()),
                "T1: expected results to be the same",
            )
            self.assertListEqual(
                list(self.ec.activities),
                list(ec_T1.activities),
                "T1: expected activities to be the same",
            )
            self.assertListEqual(
                list(self.ec.p_conn.ravel()),
                list(ec_T1.p_conn.ravel()),
                "T1: expected p_conn to be the same",
            )
            self.assertListEqual(
                list(self.ec.adj_mat.accumulator.ravel()),
                list(ec_T1.adj_mat.accumulator.ravel()),
                "T1: expected adj_mat to be the same",
            )
        finally:
            os.remove(filename)

        # run an extra few steps
        T_more = 0.05
        T2 = np.round(T + T_more, 5)
        self.ec.run_network(dt, T_more, method="Euler")
        ec_T1.run_network(dt, T_more, method="Euler")
        filename_T2 = self.ec._get_filename()
        filename_T2_new_ec = ec_T1._get_filename()

        self.assertEqual(
            filename,
            filename_T2,
            f"expected hashes at {T} and {T_more} to be the same (everything else being equal)."
            f"\nec={self.ec}\nec_T1={ec_T1}\nec_T2={ec_T2}",
        )

        self.assertEqual(
            filename_T2,
            filename_T2_new_ec,
            f"expected hashes to be the same for self and new_ec."
            f"\nec={self.ec}\nec_T1={ec_T1}\nec_T2={ec_T2}",
        )

        self.ec.save()
        try:
            self.assertFalse(
                ec_T2.load(dt, T_more),
                f"expected no results to load for T_more={T_more}",
            )
            self.assertTrue(ec_T2.load(dt, T2), f"expected results to load for T2={T2}")
            self.assertTrue(ec_T2.result is not None, "expected results to loaded")
            self.assertListEqual(
                list(self.ec.result.y.ravel()),
                list(ec_T2.result.y.ravel()),
                "T2: expected results to be the same",
            )
            self.assertListEqual(
                list(self.ec.activities),
                list(ec_T2.activities),
                "T2: expected activities to be the same",
            )
            self.assertListEqual(
                list(self.ec.p_conn.ravel()),
                list(ec_T2.p_conn.ravel()),
                "T2: expected p_conn to be the same",
            )
            self.assertListEqual(
                list(self.ec.adj_mat.accumulator.ravel()),
                list(ec_T2.adj_mat.accumulator.ravel()),
                "T2: expected adj_mat to be the same",
            )
            self.assertFalse(
                np.all(self.ec.result.y == ec_T1.result.y),
                "T2: expected results to be different when loading a simulation halfway and continuing "
                "independently (due to randomness)",
            )
        finally:
            os.remove(filename_T2)


class TestNoisyEchoChamber(TestCase):
    def setUp(self) -> None:
        self.ec = NoisyEchoChamber(1000, m=10, K=3.0, alpha=3.0)

    def test_save_load(self):
        from opdynamics.utils.distributions import negpowerlaw

        self.ec.alpha = 3  # controversialness of issue (sigmoidal shape)
        self.ec.K = 3  # social interaction strength

        dt = 0.01
        T = 0.1

        _predictable_interaction(self.ec)
        self.ec.set_dynamics(D=0.1)

        ec_D_const = NoisyEchoChamber(
            self.ec.N, self.ec.m, self.ec.K, self.ec.alpha, name=self.ec.name
        )
        _predictable_interaction(ec_D_const)
        ec_D_const.set_dynamics(D=0.1)

        filename_T0 = self.ec._get_filename()
        self.assertFalse(os.path.exists(filename_T0))
        self.assertFalse(self.ec.load(dt, T))

        self.ec.run_network(dt, T)
        filename = self.ec._get_filename()
        filename_ec_D = ec_D_const._get_filename()
        try:
            saved_filename = self.ec.save()
            self.assertTrue(saved_filename, filename)
            self.assertTrue(os.path.exists(filename))

            self.assertFalse(
                ec_D_const.has_results, "expected results to be None before loading"
            )
            self.assertTrue(ec_D_const.load(dt, T), "did not load results as expected")
            self.assertTrue(ec_D_const.has_results, "expected results to loaded")

            self.assertListEqual(
                list(self.ec.result.y.ravel()),
                list(ec_D_const.result.y.ravel()),
                "T1: expected results to be the same",
            )
            self.assertListEqual(
                list(self.ec.activities),
                list(ec_D_const.activities),
                "T1: expected activities to be the same",
            )
            self.assertListEqual(
                list(self.ec.p_conn.ravel()),
                list(ec_D_const.p_conn.ravel()),
                "T1: expected p_conn to be the same",
            )
            self.assertListEqual(
                list(self.ec.adj_mat.accumulator.ravel()),
                list(ec_D_const.adj_mat.accumulator.ravel()),
                "T1: expected adj_mat to be the same",
            )
        finally:
            os.remove(filename)

        # run an extra few steps
        T_more = 0.05
        T2 = np.round(T + T_more, 5)
        # change nosie
        self.ec.set_dynamics(D=0.5)
        self.ec.run_network(dt, T_more)
        # keep noise the same
        ec_D_const.run_network(dt, T_more)
        filename_T2 = self.ec._get_filename()
        filename_ec_D_T2 = ec_D_const._get_filename()

        self.assertNotEqual(
            filename,
            filename_T2,
            f"expected hashes at {T} and {T_more} to be different when changing noise."
            f"\n{self.ec}",
        )

        self.assertEqual(
            filename_ec_D,
            filename_ec_D_T2,
            f"expected hashes to be the same when keeping noise constant."
            f"\n{ec_D_const}",
        )

        files_to_del = [self.ec.save(), ec_D_const.save()]

        ec_D_T2 = NoisyEchoChamber(
            self.ec.N, self.ec.m, self.ec.K, self.ec.alpha, name=self.ec.name
        )
        _predictable_interaction(ec_D_T2)
        ec_D_T2.set_dynamics(D=0.1)
        filename_D_T2 = ec_D_T2._get_filename()

        try:
            self.assertEqual(
                filename_D_T2,
                filename_ec_D_T2,
                "expected same D_hist to have same filename (hash)."
                f"\n{ec_D_const}\n{ec_D_T2}",
            )
            self.assertFalse(
                ec_D_T2.has_results, "expected results to be None before loading"
            )
            self.assertTrue(ec_D_T2.load(dt, T2), "did not load results as expected")
            self.assertTrue(ec_D_T2.has_results, "expected results to loaded")

            # create D_hist to restore desired sim
            ec_D_change = NoisyEchoChamber(
                self.ec.N, self.ec.m, self.ec.K, self.ec.alpha, name=self.ec.name
            )
            _predictable_interaction(ec_D_change)
            ec_D_change.set_dynamics(D=0.1)
            ec_D_change._D_hist = [(0, 0.1), (T, 0.5)]
            filename_Dchange_T2 = ec_D_change._get_filename()
            self.assertEqual(
                filename_T2,
                filename_Dchange_T2,
                "expected same D_hist to have same filename (hash)."
                f"\n{self.ec}\n{ec_D_change}",
            )
            self.assertFalse(
                ec_D_change.has_results, "expected results to be None before loading"
            )
            self.assertTrue(
                ec_D_change.load(dt, T2), "did not load results as expected"
            )
            self.assertTrue(ec_D_change.has_results, "expected results to loaded")
        finally:
            for f in files_to_del:
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass
