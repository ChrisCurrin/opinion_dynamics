import logging
import numpy as np

from unittest import TestCase

from opdynamics.echochamber import EchoChamber

# noinspection PyUnusedName
logger = logging.getLogger("test_echochamber")


class TestEchoChamber(TestCase):
    def setUp(self) -> None:
        self.ec = EchoChamber(1000)

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
