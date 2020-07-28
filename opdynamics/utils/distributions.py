"""
Miscellaneous distributions not included in `numpy.random` or `scipy.stats`

.. code-block:: pythondecay

    import powerlaw
    from scipy.stats import powerlaw as spower

    fig, axs = plt.subplots(1, 3)
    a, xmin = gamma, epsilon
    N = 1000

    # generates random variates of power law distribution
    vrs = powerlaw.Power_Law(xmin=xmin, parameters=[a]).generate_random(N)

    # plotting the PDF estimated from variates
    bin_min, bin_max = epsilon, 1-epsilon
    bins = 10**(np.linspace(np.log10(bin_min), np.log10(bin_max), 100))
    counts, edges = np.histogram(vrs, bins, density=True)
    centers = (edges[1:] + edges[:-1])/2.

    sns.distplot(vrs, bins=1000, kde=False, ax=axs[1])

    p2 = spower.rvs(1/gamma, loc=((1-gamma)/(1-epsilon**(1-gamma))), size=N)
    sns.distplot(p2, bins=1000, kde=False, ax=axs[1])
    # sns.distplot(negpowerlaw.rvs(gamma, epsilon, 1, size=N), bins=1000, kde=False, ax=axs[1])

    axs[1].set_xlim(0, 1)

    low=epsilon
    high=1
    ag, bg = low ** (1 - gamma), high ** (1 - gamma)

    # plotting the expected PDF
    xs = np.linspace(bin_min, bin_max, 100000)
    axs[0].plot(centers, counts, '.')
    axs[0].plot(xs, [(a-1)*xmin**(a-1)*x**(-a) for x in xs], color='red', ls='-', lw=2, alpha=0.5)
    axs[0].plot(xs, [((1-a)/(1-xmin**(1-a)))*x**(-a) for x in xs], color='g', ls='--', lw=2, alpha=0.5)
    axs[2].plot(xs, [xmin * (1 - x) ** (-1/(a - 1)) for x in xs], color='r', ls='-', lw=4, alpha=0.5)
    axs[2].plot(xs, [np.power(ag + (bg - ag) * x, 1.0 / (1 - gamma)) for x in xs], color='b', ls=':', lw=4, alpha=0.5)

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')

.. note ::
    incomplete

"""
import numpy as np


class negpowerlaw(object):
    """Power-law distribution with a negative exponent, provided the minimum is not 0.

    This class acts like `scipy.stats` distributions (but does not inherit from `rv_continuous`).

    The power-law probability distribution is

    .. math ::
        p(a) = (\\gamma-1)\\cdot \\epsilon^{\\gamma-1} \\cdot a^{-\\gamma}

        p(a) = \\frac{\\gamma-1}{\\epsilon} (\\frac{a}{\\epsilon})^{-\\gamma}

    where :math:`a` is a continuous variable, :math:`\\gamma` is the power-law exponent,
    and :math:`\\epsilon` is the lower bound.

    .. note ::

        The probability distribution can be generated a number of ways:

        .. math ::
            \\epsilon \\cdot (1 - r)^{\\frac{-1}{\\gamma - 1}}

            \\left(\\epsilon^{1 - \\gamma} +
            ({a_{max}}^{1 - \\gamma} - \\epsilon^{1 - \\gamma}) * r\\right)^{\\frac{1}{1 - \\gamma}}

        where :math:`r` is uniform random number :math:`r \\sim \\mathcal{U}(0,1)` and
        :math:`{a_{max}}` is the upper bound


    """

    name = "negpowerlaw"

    @staticmethod
    def rvs(gamma: float, low=0.01, high=1.0, size=1) -> np.ndarray:
        """Sample a random variable from the distribution.

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

    def __str__(self):
        return self.name
