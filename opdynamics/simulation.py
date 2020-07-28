"""Run a full simulation of a network of agents without worrying about object details."""
import itertools
import logging
import os

import numpy as np
import pandas as pd
from functools import partial

import vaex
from tqdm import tqdm, trange
from typing import Callable, Dict, Iterable, List, Type, TypeVar, Union

from opdynamics.utils.cache import save_results
from opdynamics.networks.echochamber import EchoChamber, OpenChamber
from opdynamics.utils.distributions import negpowerlaw
from opdynamics.visualise import (
    show_periodic_noise,
    show_simulation_results,
)

logging.basicConfig(level=logging.INFO)

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
    write_mapping=True,
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
    :param write_mapping: Write to a file that maps the object's string representation and it's hash value.

    :return: Instance of EchoChamber (or a subclass)
    """
    from opdynamics.utils.cache import cache_ec

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
        cache_ec(cache, _ec, write_mapping=write_mapping)

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
    write_mapping=True,
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
    :param write_mapping: Write to a file that maps the object's string representation and it's hash value.

    :return: NoisyEchoChamber created for the simulation.
    """
    from opdynamics.utils.cache import cache_ec

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

        cache_ec(cache, nec, write_mapping=write_mapping)

    if plot_opinion:
        show_periodic_noise(nec, noise_start, noise_length, recovery, interval, num, D)

    return nec


# TODO: replace multiprocessing with await/async


def _comp_unit(
    cls, keys, values: list, cache, write_mapping: bool = True, **kwargs
) -> (EC, dict):
    """Define unit of computation to be parallelizable.

    :param values: Values to update in kwargs according. Same order as keys.
    :param write_mapping: Whether to write to a file. Defaults to `True`.
    :return: Pair of EchoChamber object, parameters used.
    """

    # create copy of kwargs
    updated_kwargs = dict(kwargs)

    base_name = updated_kwargs.pop("name", "")
    noise_start = updated_kwargs.pop("noise_start", 0)
    if noise_start > 0:
        T = updated_kwargs.pop("T", 1.0)
        noise_length = updated_kwargs.pop("noise_length", T)
        recovery = updated_kwargs.pop("recovery", 0)

    names = []

    # re-assign
    for key, value in zip(keys, values):
        updated_kwargs[key] = value
        names.append(f"{key}={value}")

    # create a name based on changed key-values
    name = base_name + ", ".join(names)

    # run according to noise_start
    if noise_start > 0:
        nec = run_periodic_noise(
            noise_start,
            noise_length,
            recovery,
            cls=cls,
            name=name,
            cache=cache,
            write_mapping=write_mapping,
            **updated_kwargs,
        )
    else:
        nec = run_params(
            cls, name=name, cache=cache, write_mapping=write_mapping, **updated_kwargs,
        )
    return nec, updated_kwargs


def run_product(
    parameters: Dict[str, Dict[str, Union[list, str]]],
    cls: Type[EC] = EchoChamber,
    cache=False,
    cache_sim=True,
    parallel: Union[bool, int] = False,
    **kwargs,
) -> Union[pd.DataFrame, List[EC]]:
    """Run a combination of variables, varying noise for each combination.

    Examples
    --------

    .. code-block:: python

        other_vars = {
            'D':{
                'range':np.round(np.arange(0.000, 0.01, 0.002), 3),
                'title': 'nudge',
            },
            'alpha':{
                'range':[0.0001, 2, 3],
                'title':'α',
            },
        }

        df = run_product(D_range, other_vars, **kwargs)


    :param parameters: A dictionary of parameters to vary. The product of each parameter's 'range' key is taken.
        See example for format.
    :param cls: Class of noise.
    :param cache: Whether to cache individual simulations (default ``False``).
    :param cache_sim: Whether to cache the ``pd.DataFrame`` used to store the simulation results (default ``True``).
    :param parallel: Run iterations serially (``False``) or in parallel (``True``). Defaults to ``False``.
        An integer can be passed to explicitly set the pool size, else it is equalt to th number of CPU cores.
        Parallel run uses ``multiprocessing`` library and is not fully tested.

    :return: DataFrame of results in tidy long-form. That is, each column is a variable and each row is an
        observation. Only opinions at the last time point are stored.
    """
    from opdynamics.utils.cache import get_cache_dir

    plot_opinion = kwargs.pop("plot_opinion", False)

    keys = list(parameters.keys())

    # # This allows a mixed-type to be passed where `range` isn't necessary.
    # verbose_other_vars = {
    #     k: {"range": v} for k, v in parameters.items() if type(v) is not dict
    # }
    # parameters.update(verbose_other_vars)

    # determine combination of parameters
    full_range = itertools.product(*[parameters[key]["range"] for key in keys])
    cache_dir = get_cache_dir()
    file_name = os.path.join(cache_dir, "noise_source.h5")
    vaex_file_name = file_name.replace(".h5", ".hdf5")

    # exclude parameter combinations that have already been saved
    if cache_sim and os.path.exists(file_name):
        logger.debug("reading from existing file...")
        run_range = set()
        # noinspection PyTypeChecker
        chunks: Iterable = pd.read_hdf(file_name, iterator=True, chunksize=10000)
        for chunk in chunks:
            run_range.update(set(chunk.groupby([*keys]).count().index))
        full_range = (x for x in full_range if x not in run_range)

    # list to store EchoChamber objects (only if not cache_sim)
    nec_list = []

    # create helper functions for running synchronously or asynchronously
    def run_sync():
        """Normal ``for`` loop over range."""
        for values in tqdm(full_range, desc="full range"):
            nec, params = _comp_unit(cls, keys, values, cache=cache, **kwargs)
            if cache_sim:
                save_results(file_name, nec, **params)
            if not (cache or cache_sim):
                # if not being run for caching, store results in a list
                nec_list.append(nec)

    def run_async(n_processes=None):
        """Create a pool of size ``n_processes`` for running multiple networks simultaneously.
        Full range is iterated over using :func:`multiprocessing.imap` to cache results to a shared DataFrame and to
        write the object -> hash mapping safely.

        :param n_processes: Number of processes to use for the pool.

        .. seealso::
            :func:`multiprocessing.Pool`

        """
        import multiprocessing as mp

        if n_processes is None:
            n_processes = mp.cpu_count()
        p = mp.Pool(n_processes)

        # change mapping to False as it may cause multi-access errors
        kwargs.pop("write_mapping", False)
        write_file_name = os.path.join(get_cache_dir(), "map.txt")

        for nec, params in p.imap(
            partial(_comp_unit, cls, keys, cache=cache, write_mapping=False),
            full_range,
        ):
            if cache and nec.save_txt is not None:
                with open(write_file_name, "a+") as write_file:
                    write_file.write(nec.save_txt)
            if cache_sim:
                save_results(file_name, nec, **params)
            if not (cache or cache_sim):
                # if not being run for caching, store results in a list
                nec_list.append(nec)
        p.close()
        p.join()

    logger.debug(f"running")
    if parallel:
        run_async(n_processes=parallel if type(parallel) is int else None)
    else:
        run_sync()
    logger.debug(f"done")

    # load all results into a DataFrame
    #   if the system does not have enough memory, the ``vaex`` library is used.
    #       the existing file is converted to a vaex-compatible format (different hdf5 implementations)
    #       if this converted file already exists, load it directly
    #       else, conversion involves chunking and exporting the file as batches of vaex-compatible mini-files.
    #           these are combined and exported as a singular file
    if cache_sim and os.path.exists(file_name):
        logger.debug("loading full DataFrame from storage")
        if os.path.exists(vaex_file_name):
            return vaex.open(vaex_file_name)
        try:
            # noinspection PyTypeChecker
            df = pd.read_hdf(file_name)
        except MemoryError:
            logger.warning(
                "Environment ran out of memory, loading data using `vaex` insatead of `pandas`. "
                "The existing file will remain the same but needs to be converted to `vaex-hdf5` format. "
                "This will take additional space and may take a while. "
            )
            # noinspection PyTypeChecker
            chunks: Iterable = pd.read_hdf(file_name, iterator=True, chunksize=50000)
            convert_dir = os.path.join(".temp", "convert")
            try:
                os.makedirs(convert_dir)
            except IOError as err:
                logger.error(
                    "Needed to create a temporary directory for conversion but failed."
                )
                raise err
            logger.debug("converting...")
            for i, chunk in enumerate(chunks):
                vaex_df = vaex.from_pandas(chunk, copy_index=False)
                vaex_df.export(f"batch_{i}.hdf5")
            df = vaex.open("batch*.hdf5")
            logger.debug(f"saving new version to {vaex_file_name}...")
            df.export(vaex_file_name)
            logger.debug("re-assigning")
            df = vaex.open(vaex_file_name)
            logger.debug("removing conversion directory")
            try:
                os.removedirs(convert_dir)
            except IOError:
                logger.error(
                    f"Tried to delete temporary conversion directory '{convert_dir}' but failed. "
                    f"Please delete manually."
                )
                pass
            logger.debug("converted")
        return df
    return nec_list


def example():
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

    _other_vars = {
        "D": {"range": np.round(np.arange(0.000, 0.01, 0.002), 3)},
        "k_steps": {"range": [1, 10, 100], "title": "k",},
    }
    run_product(_other_vars, **kwargs)
    plt.show()


if __name__ == "__main__":
    example()
