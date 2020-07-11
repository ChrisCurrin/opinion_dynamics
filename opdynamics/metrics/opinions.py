import logging
import numpy as np

# noinspection PyProtectedMember
from numpy.random._generator import Generator


def nearest_neighbours(opinions: np.ndarray, accum_adj_mat: np.ndarray) -> np.ndarray:
    """Calculate mean value of every agents' nearest neighbour.

    .. math::
            \\frac{\\sum_j a_{ij} x_j}{\\sum_j a_{ij}}

    where :math:`a_{ij}` represents the (static) adjacency matrix of the aggregated interaction network
    and :math:`\\sum_j a_{ij}` is the degree of node `i`.

    """
    close_opinions = np.sum(accum_adj_mat * opinions, axis=0)
    out_degree_i = np.sum(accum_adj_mat, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # suppress warnings about dividing by nan or 0
        nn = close_opinions / out_degree_i
    return nn


def sample_means(
    opinions: np.ndarray, sample_size: int, num_samples: int = 1, rng: Generator = None
) -> np.ndarray:
    """
    Calculate the sample means.

    Each mean is from a sample of ``sample_size`` agents.

    The number of means is the same as ``num_samples``.

    Means are taken either from ``opinions`` argument or from ``self.result.y`` at time point ``t``.

    see https://en.wikipedia.org/wiki/Central_limit_theorem

    :param opinions: Opinions to sample from. If ``None``, use ``result.y`` array.
    :param sample_size: Pick this many agents' opinions (i.e. a sample).
    :param num_samples: Number of sample to perform.
    :param rng: Random number generator to use.

    :return: Array of means (size equal to ``num_samples``).
    """

    if rng is None:
        rng = np.random
    if len(opinions.shape) == 2:
        # for scipy integration methods
        opinions = opinions.T[0]
    # create a large array of opinions (at time t) of size N times number of samples (N*n)
    n_opinions = np.tile(opinions, num_samples)
    # uniformly pick a sample n times (sample_size*n).
    n_idx = np.asarray(
        rng.uniform(0, opinions.shape[0], (sample_size, num_samples)), dtype=int
    )
    # Take mean of samples of opinions. Note the num_samples or n dimension is the same.
    means = np.mean(n_opinions[n_idx], axis=0)
    # return samples means
    return means


# noinspection NonAsciiCharacters
def distribution_modality(opinions) -> float:
    """Calculate Test of unimodality for normal distribution(s)

    .. math::
            \\frac{v - \\mu}{\\sigma} \\leq \\sqrt{\\frac{3}{5}}

    where :math:`v` is the median, :math:`\\mu`` is the mean, and :math:`\\sigma` is the standard deviation.

    see https://en.wikipedia.org/wiki/Unimodality
    see https://doi.org/10.1007/s10182-008-0057-2 

    """
    # mean, median, mode
    ν = np.median(opinions)
    μ = np.mean(opinions)
    σ = np.var(opinions)
    is_unimodal_gauss = np.abs(ν - μ) / σ <= np.sqrt(3 / 5)

    # holzmann
    pop1 = opinions[opinions < 0]
    pop2 = opinions[opinions > 0]
    μ1 = np.mean(pop1)
    μ2 = np.mean(pop2)
    σ1 = np.var(pop1)
    σ2 = np.var(pop2)
    d = np.abs(μ1 - μ2) / (2 * np.sqrt(σ1 * σ2))
    is_unimodal = d <= 1
    p = pop1.size / opinions.size
    rhs = 2 * np.log(d - np.sqrt(d ** 2 - 1)) + 2 * d * np.sqrt(d ** 2 - 1)
    is_unimodal_alt = np.abs(np.log(1 - p) - np.log(p)) >= rhs

    # Ashman's D
    #  > For a mixture of two normal distributions D > 2 is required for a clean separation of the distributions.
    a_D = np.sqrt(2) * np.abs(μ1 - μ2) / np.sqrt(σ1 ** 2 + σ2 ** 2)
    logging.debug(
        f"d = {d}"
        f" \t a_D = {a_D}"
        f"\nis_unimodal_gauss = {is_unimodal_gauss}"
        f" \t is_unimodal = {is_unimodal}"
        f" \t is_unimodal_alt = {is_unimodal_alt}"
    )

    # get the maximum
    # TODO: finish
    max_opinion_pop1 = np.argmax(np.histogram(pop1))
    max_opinion_pop2 = np.argmax(np.histogram(pop2))
    if np.abs(max_opinion_pop1 - max_opinion_pop2) > 0.5:
        return 1
    return a_D
