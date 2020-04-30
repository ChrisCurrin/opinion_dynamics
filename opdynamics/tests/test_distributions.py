"""Test custom distributions in opdynamics.utils.distributions"""
from unittest import TestCase

import numpy as np
from scipy.stats import powerlaw

from opdynamics.utils.distributions import negpowerlaw


class Testnegpowerlaw(TestCase):
    def test_rvs(self):
        # a negative value in `negpowerlaw` should be the same as the `powerlaw` from `scipy.stats` when low=0 and b=1
        arr1 = negpowerlaw.rvs_alt(-2.1, 0, 1, size=100000)
        arr2 = powerlaw.rvs(2.1, 0, 1, size=100000)

        # test same distribution by mean and variance (within 2 decimal places)
        self.assertAlmostEqual(np.mean(arr1), np.mean(arr2), 2)
        self.assertAlmostEqual(np.var(arr1), np.var(arr2), 2)
