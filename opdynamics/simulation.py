"""Run a full simulation of a network of agents without worrying about object details."""
import logging
import numpy as np
from typing import Callable, Iterable, Type, Union

from opdynamics.dynamics.echochamber import EchoChamber, NoisyEchoChamber
from opdynamics.utils.distributions import negpowerlaw

logger = logging.getLogger("simulation")


def run_params(
    cls: Type[EchoChamber] = EchoChamber,
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
    method: str = "RK45",
    plot_opinion: Union[bool, str] = False,
    lazy: bool = False,
    *sim_args,
    **sim_kwargs,
) -> Type[EchoChamber]:
    """Static method to quickly and conveniently run a simulation where the parameters differ, but the structure
        is the same (activity distribution, dynamics, etc.)"""
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
    _ec.run_network(dt=dt, t_end=T, method=method)
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
        import seaborn as sns

        fig, ax = plt.subplots(
            nrows=len(D_range), ncols=2, sharex=True, sharey="col", figsize=(8, 11)
        )

        for i, (nec, _ax) in enumerate(zip(nec_arr, ax)):
            vis = VisEchoChamber(nec)
            # 0 is first column, 1 is 2nd column
            vis.show_opinions_snapshot(ax=_ax[0], title=(i == 0))
            if i != len(nec_arr) - 1:
                _ax[0].set_xlabel("")
                _ax[1].set_xlabel("")
            _ax[0].annotate(
                vis.ec.name,
                fontsize="x-large",
                xy=(0.01, 1.05),
                xycoords="axes fraction",
                ha="left",
                va="top",
                rotation=0,
            )
        sns.despine()

    return nec_arr


def run_periodic_noise(
    noise_start,
    noise_length,
    recovery,
    interval=0,
    num=1,
    D=0.01,
    method="Euler-Maruyama",
    *args,
    **kwargs,
):
    lazy = kwargs.pop("lazy", None)
    if lazy is not None:
        logger.warning(
            "value of 'lazy' provided to run_periodic_noise is ignored (set to True)."
        )
    logger.debug(f"letting network interact without noise until {noise_start}.")
    nec: NoisyEchoChamber = run_params(
        NoisyEchoChamber, *args, T=noise_start, D=0, method=method, lazy=True, **kwargs
    )
    noiseless_time = interval * (num - 1)
    block_time = (noise_length - noiseless_time) / num
    logger.debug(
        f"noise (D={D}) will be intermittently added from {noise_start} for {noise_length} in blocks of "
        f"{block_time:.3f} with {interval} interval. A total of {num} perturbations will be done."
    )
    for i in range(num):
        nec.set_dynamics(D=D)
        nec.run_network(t_end=block_time)
        # only include a silent block of time if this wasn't the last block
        if i < num - 1:
            nec.set_dynamics(D=0)
            nec.run_network(t_end=interval)
    logger.debug(
        f"removing noise and letting network settle at {noise_start+noise_length} for {recovery}."
    )
    nec.set_dynamics(D=0)
    nec.run_network(t_end=recovery)

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
