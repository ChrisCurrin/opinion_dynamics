"""Run a full simulation of a network of agents without worrying about object details."""
import itertools
import logging
import os

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm, trange
from typing import Callable, Dict, Iterable, List, Tuple, Type, TypeVar, Union

from opdynamics.utils.constants import *
from opdynamics.dynamics.echochamber import EchoChamber, NoisyEchoChamber, OpenChamber
from opdynamics.utils.distributions import negpowerlaw
from opdynamics.visualise import (
    show_periodic_noise,
    show_simulation_range,
    show_simulation_results,
)

logger = logging.getLogger("simulation")

EC = TypeVar("EC", bound="EchoChamber")
NEC = TypeVar("NEC", bound="NoisyEchoChamber")


def run_params(
    cls: Type[EC] = EchoChamber,
    N: int = 1000,
    m: int = 10,
    K: float = 3,
    alpha: float = 2,
    beta: float = 2,
    activity: Callable = negpowerlaw,
    gamma: float = 2.1,
    epsilon: float = 1e-2,
    r: float = 0.5,
    dt: float = 0.01,
    T: float = 1.0,
    method: str = None,
    lazy: bool = True,
    cache: Union[bool, str] = True,
    plot_opinion: Union[bool, str] = False,
    *sim_args,
    **sim_kwargs,
) -> EC:
    """
    Quickly and conveniently run a simulation where the parameters differ, but the structure
            is the same (activity distribution, dynamics, etc.)

    :param cls: The type of EchoChamber class to use. E.g. OpenChamber.
    :param N: initial value: Number of agents.
    :param m: Number of other agents to interact with.
    :param K: Social interaction strength.
    :param alpha: Controversialness of issue (sigmoidal shape).
    :param beta: Power law decay of connection probability.
    :param activity: Distribution of agents' activities.
    :param gamma: Power law distribution param.
    :param epsilon: Minimum activity level with another agent.
    :param r: Probability of a mutual interaction.
    :param dt: Maximum size of time step.
    :param T: Length of simulation.
    :param method: Solver method (custom or part of scipy).
    :param lazy: Compute social interactions when needed (True) or before the network evolves with time (False).
        Note that this can lead to large RAM usage.
    :param cache: Use a cache to retrieve and save results. Saves to `.cache`.
    :param plot_opinion: Display opinions (True), display summary figure ('summary') or display multiple figures (
        'all').

    :keyword noise_source: Whether noise is external or internal (see formulations above).
            Use constants defined in `utils.constants`.
    :keyword k_steps: If `noise_source=INTERNAL_NOISE`, then choose N random agents every `k_steps`
         (default 10).

    :return: Instance of EchoChamber (or a subclass)
    """
    if method is None:
        if cls is OpenChamber:
            method = "Euler-Maruyama"
        else:
            method = "RK45"
    logger.debug(
        f"run_params for {cls.__name__} with (N={N}, m={m}, K={K}, alpha={alpha}, beta={beta}, activity "
        f"={str(activity)}(epsilon={epsilon}, gamma={gamma}), dt={dt}, T={T}, r={r}, plot_opinion="
        f"{plot_opinion}, lazy={lazy})"
    )
    logger.debug(f"additional args={sim_args}\tadditional kwargs={sim_kwargs}")
    _ec = cls(N, m, K, alpha, *sim_args, **sim_kwargs)
    _ec.set_activities(activity, gamma, epsilon, 1, dim=1, **sim_kwargs)
    _ec.set_connection_probabilities(beta=beta, **sim_kwargs)
    _ec.set_social_interactions(r=r, lazy=lazy, dt=dt, t_end=T)
    _ec.set_dynamics(*sim_args, **sim_kwargs)
    if not (cache and _ec.load(dt, T)):
        _ec.run_network(dt=dt, t_end=T, method=method)
        if cache:
            _ec.save(cache != "all")
    if plot_opinion:
        show_simulation_results(_ec, plot_opinion)
    return _ec


def run_periodic_noise(
    noise_start: float,
    noise_length: float,
    recovery: float,
    interval: float = 0.0,
    num: int = 1,
    cls: Type[NEC] = OpenChamber,
    D: float = 0.01,
    N: int = 1000,
    m: int = 10,
    K: float = 3,
    alpha: float = 2,
    beta: float = 2,
    activity: Callable = negpowerlaw,
    gamma: float = 2.1,
    epsilon: float = 1e-2,
    r: float = 0.5,
    dt: float = 0.01,
    cache: bool = True,
    method: str = "Euler-Maruyama",
    plot_opinion: bool = False,
    *args,
    **kwargs,
) -> NEC:
    """
    Run a simulation with no noise, then bursts/periods of noise, then a recovery period.

    no noise | noise | recovery

    _________|-------|_________

    The frequency at which noise is applied can be optionally set using ``interval`` and ``num``.

    Examples
    ----
    To have 5 s no noise, 10 s constant noise, and 7 s recovery:

    .. code-block :: python

        run_periodic_noise(noise_start=5, noise_length=10, recovery=7)

    To have 5 s no noise, 10 s with periodic noise, and 7 s recovery

    with the block of noise defined as 2 periods of noise separated by 0.5 s of no noise.

    _____|---- ----|_______

        .. code-block :: python

            run_periodic_noise(noise_start=5, noise_length=10, recovery=7, num=2, interval=0.5)

        To have 5 s no noise, 10 s with periodic noise, and 7 s recovery

    with the block of noise defined as 5 periods of noise each separated by 0.5 s of no noise.

    noise per block = 10s/5 - 0.5s = 1.5 s

    _____|- - - - -|_______

        .. code-block :: python

            run_periodic_noise(noise_start=5, noise_length=10, recovery=7, num=5, interval=0.5)

    :param noise_start: The time to start the noise.
    :param noise_length: The duration which will have noise. Note, this is inclusive of interval durations.
    :param recovery: Duration **after** noise is applied where there is no noise.
    :param interval: Duration between noise blocks.
    :param num: Number of noise blocks.
    :param cls: Class of the noise type to use. Must support 'D' argument.
    :param D: Noise strength.

    :param N: initial value: Number of agents.
    :param m: Number of other agents to interact with.
    :param alpha: Controversialness of issue (sigmoidal shape).
    :param K: Social interaction strength.
    :param beta: Power law decay of connection probability.
    :param activity: Distribution of agents' activities.
    :param gamma: Power law distribution param.
    :param epsilon: Minimum activity level with another agent.
    :param r: Probability of a mutual interaction.
    :param dt: Maximum size of time step.
    :param method: Solver method (custom or part of scipy).
    :param cache: Use a cache to retrieve and save results. Saves to `.cache`.

    :param plot_opinion: Whether to display results (default False).

    :keyword noise_source: Whether noise is external or internal (see formulations above).
            Use constants defined in `utils.constants`.
    :keyword k_steps: If `noise_source=INTERNAL_NOISE`, then choose N random agents every `k_steps`
         (default 10).

    :return: NoisyEchoChamber created for the simulation.
    """
    lazy = kwargs.pop("lazy", True)
    if not lazy:
        logger.warning(
            "value of 'lazy' provided to run_periodic_noise is ignored (set to True)."
        )
        lazy = True
    logger.debug(f"letting network interact without noise until {noise_start}.")
    noiseless_time = interval * (num - 1)
    block_time = (noise_length - noiseless_time) / num
    logger.debug(
        f"noise (D={D}) will be intermittently added from {noise_start} for {noise_length} in blocks of "
        f"{block_time:.3f} with intervals of {interval}. A total of {num} perturbations will be done."
    )
    t = trange(
        int(noise_start > 0) + (num * 2 - 1) + int(recovery > 0), desc="periodic noise"
    )
    name = kwargs.pop("name", "")
    name += f"[num={num} interval={interval}]"
    nec = cls(N, m, K, alpha, *args, **kwargs)
    nec.set_activities(activity, gamma, epsilon, 1, dim=1)
    nec.set_connection_probabilities(beta=beta)
    nec.set_social_interactions(r=r, lazy=lazy)
    nec.set_dynamics(D=0, *args, **kwargs)

    # try to hit the cache by creating noise history
    total_time = 0.0
    nec._D_hist = [(total_time, 0.0)]
    for i in range(num):
        total_time += block_time
        nec._D_hist.append((total_time, D))
        total_time += block_time
        nec._D_hist.append((total_time, 0.0))

    if not cache or (cache and not nec.load(dt, noise_start + noise_length + recovery)):
        # reset history
        nec._D_hist = []
        if noise_start > 0:
            nec.set_dynamics(D=0, *args, **kwargs)
            nec.run_network(dt=dt, t_end=noise_start, method=method)
            t.update()
        # inner loop of noise on-off in blocks
        for i in trange(
            num, desc="noise blocks", disable=logger.getEffectiveLevel() <= logging.INFO
        ):
            nec.set_dynamics(D=D)
            nec.run_network(t_end=block_time)
            t.update()
            # only include a silent block of time if this wasn't the last block
            if i < num - 1:
                nec.set_dynamics(D=0)
                nec.run_network(t_end=interval)
                t.update()
        logger.debug(
            f"removing noise and letting network settle at {noise_start + noise_length} for {recovery}."
        )
        if recovery > 0:
            nec.set_dynamics(D=0)
            nec.run_network(t_end=recovery)
            t.update()
        if cache:
            nec.save(cache != "all")
    if plot_opinion:
        show_periodic_noise(nec, noise_start, noise_length, recovery, interval, num, D)

    return nec


def run_noise_range(
    D_range: Iterable,
    *args,
    cls: Type[NEC] = OpenChamber,
    noise_start=0,
    plot_opinion: Union[bool, Tuple[Figure, Axes]] = True,
    **kwargs,
) -> List[NEC]:
    """ Run the same simulation multiple times, with different noises.

    :param D_range: List of noise values (`D`).
    :param cls: Class of the noise type to use. Must support 'D' argument.
    :param noise_start: Time to start adding noise.
        If 0, ``run_params`` is called, otherwise ``run_periodic_noise`` is called.
    :param plot_opinion: Display simulation range results (True). If a tuple of (fig, ax), then the passed values are used.

    :keyword N: initial value: Number of agents.
    :keyword m: Number of other agents to interact with.
    :keyword alpha: Controversialness of issue (sigmoidal shape).
    :keyword K: Social interaction strength.
    :keyword epsilon: Minimum activity level with another agent.
    :keyword gamma: Power law distribution param.
    :keyword beta: Power law decay of connection probability.
    :keyword r: Probability of a mutual interaction.
    :keyword dt: Maximum size of time step.
    :keyword T: Length of simulation.
    :keyword method: Solver method (custom or part of scipy).
    :keyword lazy: Compute social interactions when needed (True) or before the network evolves with time (False).
        Note that this can lead to large RAM usage.
    :keyword cache: Use a cache to retrieve and save results. Saves to `.cache`.

    :return: List of NoisyEchoChambers.
    """
    from tqdm.contrib import tenumerate

    nec_arr = []

    name = kwargs.pop("name", "")
    T = kwargs.pop("T", 1.0)
    noise_length = kwargs.pop("noise_length", T)
    recovery = kwargs.pop("recovery", 0)

    for i, D in tenumerate(D_range, desc=f"noise range [{name}]"):
        if noise_start > 0:
            nec = run_periodic_noise(
                noise_start,
                noise_length,
                recovery,
                *args,
                cls=cls,
                D=D,
                name=f"D={D}",
                **kwargs,
            )
        else:
            nec = run_params(cls, *args, D=D, name=f"D={D}", T=T, **kwargs)
        nec_arr.append(nec)

    if plot_opinion:
        show_simulation_range(D_range, nec_arr, plot_opinion)

    return nec_arr


def run_noise_other_range(
    D_range: Iterable,
    other_var: str,
    other_range: Iterable,
    *args,
    plot_opinion: bool = True,
    title: str = "",
    label_precision: int = None,
    subplot_kws: dict = None,
    **kwargs,
) -> (List[List[NEC]], pd.DataFrame):

    if plot_opinion:
        import matplotlib.pyplot as plt

        if subplot_kws is None:
            subplot_kws = {}
        # default to share x and y
        subplot_kws = {**dict(sharex="all", sharey="all"), **subplot_kws}
        # noinspection PyTypeChecker
        fig, ax = plt.subplots(
            nrows=len(D_range), ncols=len(other_range), squeeze=False, **subplot_kws
        )

    nec_arrs = []
    df_builder = []
    for i, other in tqdm(enumerate(other_range), desc=other_var):
        kwargs[other_var] = other
        if plot_opinion:
            other_val = (
                f"{other}"
                if label_precision is None
                else f"{np.round(other, label_precision)}"
            )
            # noinspection PyUnboundLocalVariable
            nec_arr = run_noise_range(
                D_range,
                *args,
                plot_opinion=(fig, ax[:, i]),
                name=f"{other_var}={other_val}",
                **kwargs,
            )
            ax[0, i].set_title("")
            ax[-1, i].set_xlabel(other_val)
        else:
            nec_arr = run_noise_range(D_range, plot_opinion=False, **kwargs)
        nec_arrs.append(nec_arr)

        # put data into dictionaries with keys for column names
        for nec, D in zip(nec_arr, D_range):
            for y_idx, opinion in enumerate(nec.result.y[:, -1]):
                d = {"D": D, "i": y_idx, "opinion": opinion, other_var: other}
                df_builder.append(d)

    df = pd.DataFrame(df_builder)

    if plot_opinion:
        from matplotlib.cbook import flatten
        import seaborn as sns

        fig.suptitle(title)
        fig.subplots_adjust(hspace=-0.5, wspace=0)
        for _ax in flatten(ax[:, 1:]):
            _ax.set_ylabel("", ha="right")
        for i in range(len(D_range)):
            ax[i, 0].set_ylabel(f"{D_range[i]}", ha="right")
        for _ax in fig.axes:
            _ax.set_facecolor("None")
            sns.despine(ax=_ax, bottom=True, left=True)
            _ax.tick_params(
                bottom=False, left=False, labelleft=False, labelbottom=False
            )
    return nec_arrs, df


def run_product(
    other_vars: Dict[str, Dict[str, Union[list, str]]],
    cls: Type[NEC] = OpenChamber,
    cache=True,
    cache_sim=False,
    **kwargs,
) -> pd.DataFrame:
    """Run a combination of variables, varying noise for each combination.

    Examples
    --------

    .. code-block:: python

        other_vars = {
            'D':{
                'range':np.round(np.arange(0.000, 0.01, 0.002), 3),
                'title': 'nudge',
            },
            'k_steps':{
                'range':[1, 10, 100],
                'title':'k',
            },
        }

        df = run_product(D_range, other_vars, **kwargs)


    :param other_vars: A dictionary of parameters to vary. The product of each parameter's 'range' key is taken.
        See example for format.
    :param cls: Class of noise.
    :param cache: Whether to cache the Dataframe used to store the results (default ``True``).
    :param cache_sim: Whether to cache individual simulations (default ``False``).

    :return: DataFrame of results in tidy long-form. That is, each column is a variable and each row is an
        observation. Only opinions at the last time point are stored.
    """
    from tqdm.contrib import tenumerate

    keys = list(other_vars.keys())

    full_range = list(itertools.product(*[other_vars[key]["range"] for key in keys]))
    file_name = os.path.join(".cache", "noise_source.h5")

    # the efficient HDF format is used for saving and loading DataFrames.
    if cache and os.path.exists(file_name):
        # noinspection PyTypeChecker
        df: pd.DataFrame = pd.read_hdf(file_name, columns=keys, iterator=True)
        run_range = set(df.groupby([*keys]).count().index)
        full_range = {x for x in full_range if x not in run_range}
    else:
        df = pd.DataFrame()
    df_builder = []

    kw_name = kwargs.pop("name", "")
    T = kwargs.pop("T", 1.0)
    noise_start = kwargs.pop("noise_start", 0)
    noise_length = kwargs.pop("noise_length", T)
    recovery = kwargs.pop("recovery", 0)

    for i, values in tenumerate(full_range, desc="full range"):
        names = []
        for key, value in zip(keys, values):
            kwargs[key] = value
            names.append(f"{key}={value}")
        name = kw_name + ", ".join(names)
        if noise_start > 0:
            nec = run_periodic_noise(
                noise_start,
                noise_length,
                recovery,
                cls=cls,
                name=name,
                cache=cache_sim,
                **kwargs,
            )
        else:
            nec = run_params(cls, name=name, cache=cache_sim, T=T, **kwargs)

        # put data into dictionaries with keys for column names
        for y_idx, opinion in enumerate(nec.result.y[:, -1]):
            d = {"i": y_idx, "opinion": opinion, **kwargs}
            df_builder.append(d)

    df = pd.concat([df, pd.DataFrame(df_builder)], ignore_index=True)

    if cache:
        df.to_hdf(file_name, "df")

    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _kwargs = dict(
        N=1000,  # number of agents
        m=10,  # number of other agents to interact with
        alpha=2,  # controversialness of issue (sigmoidal shape)
        K=3,  # social interaction strength
        epsilon=1e-2,  # minimum activity level with another agent
        gamma=2.1,  # power law distribution param
        beta=2,  # power law decay of connection probability
        activity_distribution=negpowerlaw,
        r=0.5,
        dt=0.01,
        t_end=0.5,
    )

    # run_params(
    #     EchoChamber, **_kwargs, plot_opinion="summary",
    # )
    kwargs = dict(
        N=1000,
        m=10,
        T=10,
        epsilon=1e-2,
        gamma=2.1,
        dt=0.01,
        K=2,
        beta=1,
        alpha=3,
        r=0.65,
        k_steps=10,
    )

    _D_range = np.round(np.arange(0.000, 0.01, 0.002), 3)

    _other_vars = {
        "k_steps": {"range": [1, 10, 100], "title": "k",},
    }
    run_product(_D_range, _other_vars, **kwargs)
    plt.show()
