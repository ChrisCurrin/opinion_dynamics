"""Miscellaneous distributions not included in `numpy.random` or `scipy.stats`"""
import numpy as np


class negpowerlaw(object):
    """Power law distribution with a negative exponent, provided the minimum is not 0.

    Class that acts like `scipy.stats` distributions (but does not inherit from `rv_continuous`).
    """

    @staticmethod
    def rvs(gamma: float, low=0.01, high=1.0, size=1) -> np.ndarray:
        """Power-law for a negative exponent.

        see https://stackoverflow.com/questions/17882907/python-scipy-stats-powerlaw-negative-exponent

        :param gamma: the *negative* power of the exponent.
        :param low: lower bound of distribution values (inclusive).
        :param high: upper bound of distribution values (inclusive).
        :param size: how many samples to take.

        :return: Power law-distributed values, in the range [low, high], with length of `size`.
        """
        r = np.random.random(size=size)
        ag, bg = low ** (1 - gamma), high ** (1 - gamma)
        return np.power(ag + (bg - ag) * r, 1.0 / (1 - gamma))

    @staticmethod
    def rvs_alt(gamma: float, low=0.01, high=1.0, size=1) -> np.ndarray:
        """Power-law gen for pdf(x) proportional to x^{gamma-1} for low<=x<=high.

        see https://stackoverflow.com/a/31117560/5209000

        Dev notes:
            - this is very similar to the `.rvs` method above, but produces slightly different results which are
                farther off from the figures in the source paper [Baumann et al. 2019].

        :param gamma: the *negative* power of the exponent.
            that is a negative value for gamma is the same as `scipy.stats.powerlaw.rvs(gamma)`.
        :param low: lower bound of distribution values (inclusive).
        :param high: upper bound of distribution values (inclusive).
        :param size: how many samples to take.

        :return: Power law-distributed values, in the range [low, high], with length of `size`.
        """
        r = np.random.random(size=size)
        ag, bg = low ** -gamma, high ** -gamma
        return np.power(ag + (bg - ag) * r, 1.0 / -gamma)

    @staticmethod
    def pdf(x, a, b, g) -> np.ndarray:
        ag, bg = a ** g, b ** g
        return g * x ** (g - 1) / (bg - ag)
