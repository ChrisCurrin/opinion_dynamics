import logging
from typing import Callable, Type, Union

from opdynamics.echochamber import EchoChamber
from opdynamics.utils.distributions import negpowerlaw

logger = logging.getLogger("simulation")


class Simulation(object):
    """Run a full simulation of a network of agents without worrying about object details."""

    @staticmethod
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
    ):
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
            from opdynamics.visualise import VisEchoChamber
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kwargs = dict(
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

    Simulation.run_params(
        EchoChamber, **kwargs, plot_opinion=True,
    )
    plt.show()
