"""Run a full simulation of a network of agents without worrying about object details."""
import logging
import numpy as np
from typing import Callable, Iterable, Type, TypeVar, Union

from tqdm import tqdm, trange

from opdynamics.dynamics.echochamber import EchoChamber, NoisyEchoChamber
from opdynamics.utils.distributions import negpowerlaw

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
    epsilon: float = 1e-2,
    gamma: float = 2.1,
    dt: float = 0.01,
    T: float = 1.0,
    r: float = 0.5,
    method: str = None,
    plot_opinion: Union[bool, str] = False,
    lazy: bool = True,
    cache: bool = True,
    *sim_args,
    **sim_kwargs,
) -> EC:
    """Static method to quickly and conveniently run a simulation where the parameters differ, but the structure
        is the same (activity distribution, dynamics, etc.)"""
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
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        from opdynamics.visualise.visechochamber import VisEchoChamber
        import seaborn as sns

        logger.debug("plotting")
        _vis = VisEchoChamber(_ec)
        if plot_opinion == "summary":
            fig, ax = _vis.show_summary(single_fig=True)
            fig.subplots_adjust(wspace=0.5, hspace=0.3, top=0.95, right=0.9)
            sns.despine()
        elif plot_opinion == "all":
            _vis.show_summary(single_fig=False)
            sns.despine()
        else:
            _vis.show_opinions(True)
    return _ec


def run_noise_range(D_range: Iterable, *args, lazy=True, plot_opinion=True, **kwargs):
    nec_arr = []

    for D in D_range:
        nec = run_params(
            NoisyEchoChamber, *args, lazy=lazy, D=D, name=f"D={D}", **kwargs,
        )
        nec_arr.append(nec)

    if plot_opinion:
        from opdynamics.visualise.visechochamber import VisEchoChamber
        import matplotlib.pyplot as plt
        import seaborn as sns

        cs = sns.color_palette("husl", n_colors=len(D_range))
        fig, ax = plt.subplots(
            nrows=len(D_range), ncols=1, sharex="all", sharey="col", figsize=(8, 11)
        )
        for i, (nec, _ax) in enumerate(zip(nec_arr, ax)):
            vis = VisEchoChamber(nec)
            # 0 is first column, 1 is 2nd column
            vis.show_opinions_snapshot(ax=_ax, title=(i == 0), color=cs[i])
            if i != len(nec_arr) - 1:
                _ax.set_xlabel("")
            _ax.set_ylabel(
                vis.ec.name,
                color=cs[i],
                fontsize="x-large",
                rotation=0,
                va="top",
                ha="left",
            )
            # _ax.annotate(
            #     vis.ec.name,
            #     fontsize="x-large",
            #     xy=(0.01, 1.05),
            #     xycoords="axes fraction",
            #     ha="left",
            #     va="top",
            #     rotation=0,
            # )
        sns.despine()

    return nec_arr


def run_periodic_noise(
    noise_start: float,
    noise_length: float,
    recovery: float,
    interval: float = 0.0,
    num: int = 1,
    D: float = 0.01,
    method: str = "Euler-Maruyama",
    plot_opinion: bool = False,
    *args,
    **kwargs,
):
    lazy = kwargs.pop("lazy", None)
    if lazy is not None:
        logger.warning(
            "value of 'lazy' provided to run_periodic_noise is ignored (set to True)."
        )
    logger.debug(f"letting network interact without noise until {noise_start}.")
    t = trange(2 + num * 2, desc="periodic noise")
    nec = run_params(
        NoisyEchoChamber, *args, T=noise_start, D=0, method=method, lazy=True, **kwargs
    )
    t.update()
    noiseless_time = interval * (num - 1)
    block_time = (noise_length - noiseless_time) / num
    logger.debug(
        f"noise (D={D}) will be intermittently added from {noise_start} for {noise_length} in blocks of "
        f"{block_time:.3f} with intervals of {interval}. A total of {num} perturbations will be done."
    )
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

    if plot_opinion:
        logger.debug("plotting periodic noise")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import gridspec
        from opdynamics.visualise import VisEchoChamber

        vis = VisEchoChamber(nec)

        # create figure and axes
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(nrows=3, ncols=3, figure=fig, wspace=0.3, hspace=0.8)
        ax_time = fig.add_subplot(gs[0:2, :])
        ax_start = fig.add_subplot(gs[-1, 0])
        ax_noise = fig.add_subplot(gs[-1, 1], sharey=ax_start)
        ax_recovery = fig.add_subplot(gs[-1, 2], sharey=ax_start)

        _colors = sns.color_palette("husl")

        # plot graphs
        vis.show_opinions(ax=ax_time, color_code="line", subsample=5, title=False)
        vis.show_opinions_snapshot(
            ax=ax_start, t=noise_start, title=f"t={noise_start}", color=_colors[0]
        )
        vis.show_opinions_snapshot(
            ax=ax_noise,
            t=noise_start + noise_length,
            title=f"t={noise_start + noise_length}",
            color=_colors[1],
        )
        vis.show_opinions_snapshot(
            ax=ax_recovery,
            t=-1,
            title=f"t={noise_start + noise_length + recovery}",
            color=_colors[2],
        )

        # adjust view limits
        from scipy import stats

        x_data, y_data = nec.result.t, nec.result.y
        s = stats.describe(y_data)
        lower_bound, upper_bound = s.mean - s.variance, s.mean + s.variance
        mask = np.logical_and(lower_bound < y_data, y_data < upper_bound)
        y_mask = y_data[mask]
        lim = (np.min(y_mask), np.max(y_mask))
        ax_time.set_ylim(*lim)
        ax_start.set_xlim(*lim)
        ax_noise.set_xlim(*lim)
        ax_recovery.set_xlim(*lim)

        # annotate plots

        # points where opinion snapshots are taken
        ax_time.vlines(
            x=[
                noise_start,
                noise_start + noise_length,
                noise_start + noise_length + recovery,
            ],
            ymin=lim[0],
            ymax=lim[1],
            color=_colors,
        )

        # noise on/off
        block_times_s = [
            noise_start + block_time * i + interval * i for i in range(num + 1)
        ]
        block_times_e = [
            noise_start + block_time * (i + 1) + interval * i for i in range(num + 1)
        ]
        ax_time.hlines(
            y=[lim[1]] * num, xmin=block_times_s, xmax=block_times_e, lw=10, color="k",
        )

        # value of noise
        ax_time.annotate(
            f"noise = {D}", xy=(noise_start, lim[1]), ha="left", va="bottom"
        )
        ax_time.annotate(
            f"noise = 0",
            xy=(noise_start + noise_length, lim[1]),
            ha="left",
            va="bottom",
        )

        sns.despine()
        ax_noise.set_ylabel("")
        ax_recovery.set_ylabel("")

    return nec


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    _kwargs = dict(
        num_agents=1000,
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
