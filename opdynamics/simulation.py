"""Run a full simulation of a network of agents without worrying about object details."""
import logging
from typing import Callable, Iterable, List, Tuple, Type, TypeVar, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm, trange

from opdynamics.dynamics.echochamber import EchoChamber, NoisyEchoChamber
from opdynamics.utils.distributions import negpowerlaw
from opdynamics.visualise import (
    show_periodic_noise,
    show_simulation_range,
    show_simulation_results,
)

logger = logging.getLogger("simulation")

EC = TypeVar("EC", bound="EchoChamber")


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
    cache: bool = True,
    plot_opinion: Union[bool, str] = False,
    *sim_args,
    **sim_kwargs,
) -> EC:
    """
    Static method to quickly and conveniently run a simulation where the parameters differ, but the structure
            is the same (activity distribution, dynamics, etc.)
    :param cls: The type of EchoChamber class to use. E.g. NoisyEchoChamber.
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
    :return: Instance of EchoChamber (or a subclass)
    """
    if method is None:
        if cls is EchoChamber:
            method = "RK45"
        elif cls is NoisyEchoChamber:
            method = "Euler-Maruyama"
    logger.debug(
        f"run_params for {cls.__name__} with (N={N}, m={m}, K={K}, alpha={alpha}, beta={beta}, activity "
        f"={str(activity)}(epsilon={epsilon}, gamma={gamma}), dt={dt}, T={T}, r={r}, plot_opinion="
        f"{plot_opinion}, lazy={lazy})"
    )
    logger.debug(f"additional args={sim_args}\tadditional kwargs={sim_kwargs}")
    _ec = cls(N, m, K, alpha, *sim_args, **sim_kwargs)
    _ec.set_activities(activity, gamma, epsilon, 1, dim=1)
    _ec.set_connection_probabilities(beta=beta)
    _ec.set_social_interactions(r=r, lazy=lazy, dt=dt, t_end=T)
    _ec.set_dynamics(*sim_args, **sim_kwargs)
    if not cache or (cache and not _ec.load(dt, T)):
        _ec.run_network(dt=dt, t_end=T, method=method)
        if cache:
            _ec.save()
    if plot_opinion:
        show_simulation_results(_ec, plot_opinion)
    return _ec


def run_noise_range(
    D_range: Iterable,
    *args,
    plot_opinion: Union[bool, Tuple[Figure, Axes]] = True,
    **kwargs,
) -> List[NoisyEchoChamber]:
    """ Run the same simulation multiple times, with different noises.

    :param D_range: List of noise values (`D`).
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
    nec_arr = []
    name = kwargs.pop("name", "")
    for D in D_range:
        nec = run_params(NoisyEchoChamber, *args, D=D, name=f"D={D} {name}", **kwargs)
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
    label_precision: str = None,
    subplot_kws: dict = None,
    **kwargs,
) -> List[List[NoisyEchoChamber]]:

    if plot_opinion:
        import matplotlib.pyplot as plt

        if subplot_kws is None:
            subplot_kws = {}
        # default to share x and y
        subplot_kws = {**dict(sharex="all", sharey="all"), **subplot_kws}
        # noinspection PyTypeChecker
        fig, ax = plt.subplots(
            nrows=len(D_range), ncols=len(other_range), **subplot_kws
        )

    nec_arrs = []

    for i, other in tqdm(enumerate(other_range)):
        kwargs[other_var] = other
        if plot_opinion:
            other_val = (
                f"{other}"
                if label_precision is None
                else f"{{}}:.{label_precision}f".format(other)
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
    return nec_arrs


def run_periodic_noise(
    noise_start: float,
    noise_length: float,
    recovery: float,
    interval: float = 0.0,
    num: int = 1,
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
    T: float = 1.0,
    cache: bool = True,
    method: str = "Euler-Maruyama",
    plot_opinion: bool = False,
    *args,
    **kwargs,
) -> NoisyEchoChamber:
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
    :param T: Length of simulation.
    :param method: Solver method (custom or part of scipy).
    :param cache: Use a cache to retrieve and save results. Saves to `.cache`.

    :param plot_opinion: Whether to display results (default False).

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
    t = trange(2 + num * 2, desc="periodic noise")
    nec = NoisyEchoChamber(N, m, K, alpha, *args, **kwargs)
    nec.set_activities(activity, gamma, epsilon, 1, dim=1)
    nec.set_connection_probabilities(beta=beta)
    nec.set_social_interactions(r=r, lazy=lazy)
    nec.set_dynamics(D=D, *args, **kwargs)

    # try to hit the cache by creating noise history
    total_time = 0.0
    nec._D_hist = [(total_time, 0.0)]
    for i in range(num):
        total_time += block_time
        nec._D_hist.append((total_time, D))
        total_time += block_time
        nec._D_hist.append((total_time, 0.0))

    if not cache or (cache and not nec.load(dt, T)):
        nec._D_hist = [(0, 0)]
        nec.run_network(dt=dt, t_end=T, method=method)

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
        nec.set_dynamics(D=0)
        nec.run_network(t_end=recovery)
        t.update()
        if cache:
            nec.save()
    if plot_opinion:
        show_periodic_noise(nec, noise_start, noise_length, recovery, interval, num, D)

    return nec


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

    run_params(
        EchoChamber, **_kwargs, plot_opinion="summary",
    )
    plt.show()
