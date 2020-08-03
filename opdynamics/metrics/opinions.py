import itertools
import logging
import os

import json
import numpy as np
import pandas as pd

# noinspection PyProtectedMember
import vaex
from numpy.random._generator import Generator

from opdynamics.utils.cache import get_cache_dir
from opdynamics.utils.decorators import hash_repeat
from opdynamics.utils.plot_utils import df_multi_mask

logger = logging.getLogger("opinion metrics")


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
def _basic_unimodal(opinions: np.ndarray) -> bool:
    """Check if a Gaussian distribution using

    .. math::
         \\frac{v - \\mu}{\\sigma} \\leq \\sqrt{\\frac{3}{5}}

    where :math:`v` is the median, :math:`\\mu`` is the mean, and :math:`\\sigma` is the standard deviation.


    """
    ν = np.median(opinions)
    μ = np.mean(opinions)
    σ = np.var(opinions)
    return np.abs(ν - μ) / σ <= np.sqrt(3 / 5)


# noinspection NonAsciiCharacters
def _holzmann_unimodal(opinions) -> bool:
    """
    see https://en.wikipedia.org/wiki/Unimodality
    see https://doi.org/10.1007/s10182-008-0057-2 
    """
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
    return is_unimodal and is_unimodal_alt


# noinspection NonAsciiCharacters
def _ashmans_d(opinions: np.ndarray) -> float:
    """Ashman's D

    For a mixture of two normal distributions D > 2 is required for a clean separation of the distributions.

    """
    pop1 = opinions[opinions < 0]
    pop2 = opinions[opinions > 0]

    μ1 = np.mean(pop1)
    μ2 = np.mean(pop2)
    σ1 = np.var(pop1)
    σ2 = np.var(pop2)
    return np.sqrt(2) * np.abs(μ1 - μ2) / np.sqrt(σ1 ** 2 + σ2 ** 2)


# noinspection NonAsciiCharacters
def distribution_modality(opinions: np.ndarray, bin_width: float = 0.1) -> float:
    """Determine the distance between population density peaks.
    Closer to 0 indicates a unimodal distribution.

    """
    pop1 = opinions[opinions < 0]
    pop2 = opinions[opinions > 0]

    # 10 % weighted threshold
    thresh = int(opinions.size * 0.1)
    if pop1.size <= thresh or pop2.size <= thresh:
        # return the negative absolute mean value for the radicalised population
        return -1 * np.max([np.abs(np.mean(pop1)), np.abs(np.mean(pop2))])

    hist, bin_edges = np.histogram(
        pop1,
        bins=np.round(np.arange(np.floor(np.min(pop1)), 0.01, bin_width), 1),
        range=(np.floor(np.min(pop1)), 0),
    )
    max_opinion_pop1 = bin_edges[1 + np.argmax(hist)]
    hist, bin_edges = np.histogram(
        pop2,
        bins=np.round(np.arange(0, np.ceil(np.max(pop2)), bin_width), 1),
        range=(0, np.ceil(np.max(pop2))),
    )
    max_opinion_pop2 = bin_edges[1 + np.argmax(hist)]
    return max_opinion_pop2 - max_opinion_pop1


def calc_distribution_differences(
    data: pd.DataFrame, x: str, y: str, variables: dict, N: int = 1000, **kwargs
):

    x_range = variables[x]["range"] if "range" in variables[x] else variables[x]
    y_range = variables[y]["range"] if "range" in variables[y] else variables[y]

    z_vars = {k: v for k, v in variables.items() if k != x and k != y}
    zs = pd.DataFrame()
    keys = list(z_vars.keys())
    value_combinations = list(
        itertools.product(*[z_vars[key]["range"] for key in keys])
    )

    for i, values in enumerate(value_combinations):
        z = mask_and_metric(data, keys, values, x, y, x_range, y_range, N, **kwargs)

        # mean across y range
        comp = z.mean(axis="columns").reset_index()
        for key, value in zip(keys, values):
            comp.loc[:, key] = value
        zs = pd.concat([zs, comp])
    zs = zs.rename(columns={"index": "D", 0: "|peak distance|"})
    return zs


def mask_and_metric(data, keys, values, x, y, x_range, y_range, N, **kwargs):
    default_kwargs = {k: v for k, v in zip(keys, values)}
    desc = json.dumps(default_kwargs, sort_keys=True)
    logger.debug(f"{desc}")
    df = df_multi_mask(data, default_kwargs)
    cache_dir = get_cache_dir()
    file_name = os.path.join(
        cache_dir, f"{hash_repeat({x: x_range, y: y_range, **default_kwargs})}.h5"
    )
    if os.path.exists(file_name):
        logger.debug(f"\t load")
        z = pd.read_hdf(file_name)
    else:
        z = pd.DataFrame(index=x_range, columns=y_range, dtype=np.float64)
        if isinstance(df, pd.DataFrame):
            for x_val, y_val in itertools.product(x_range, y_range):
                opinions = df_multi_mask(df, {x: x_val, y: y_val})["opinion"]
                z.loc[x_val, y_val] = distribution_modality(opinions, **kwargs)
        else:
            for start, stop, arrs in df.sort([x, y]).to_arrays(
                column_names=[x, y, "opinion"], chunk_size=N
            ):
                _xs, _ys, opinions = arrs
                x_check = all(_xs == _xs[0])
                y_check = all(_ys == _ys[0])
                print(f"{start}-{stop}", end="\r")
                if not (x_check and y_check):
                    raise IndexError(
                        f"expected all {x} and {y} values to be the same for chunk sizes of {N}."
                        f"\n{_xs} # {_ys}"
                    )
                z.loc[_xs[0], _ys[0]] = distribution_modality(opinions, **kwargs)
        z.to_hdf(file_name, key="df")
        logger.debug(f"\t saved")
    return z
