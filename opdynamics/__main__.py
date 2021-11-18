"""Run opdynamics using the command line with `python -m opdynamics`"""
import argparse
import logging

# noinspection PyProtectedMember
from scipy.integrate._ivp.ivp import METHODS as SCIPY_METHODS

from opdynamics.socialnetworks import (
    ConnChamber,
    ContrastChamber,
    SocialNetwork,
    OpenChamber,
    SampleChamber,
)
from opdynamics.simulation import run_product
from opdynamics.integrate.solvers import ODE_INTEGRATORS, SDE_INTEGRATORS
from opdynamics.utils.distributions import negpowerlaw
from opdynamics.visualise import show_simulation_results


class Formatter(argparse.HelpFormatter):
    """Change help format to display 'usage' according to the defined argument order
    (instead of positional elements last).

    see https://stackoverflow.com/a/26986546
    """

    def _format_usage(self, usage, actions, groups, prefix):
        if prefix is None:
            prefix = "usage: "

        # if usage is specified, use that
        if usage is not None:
            # noinspection PyAugmentAssignment
            usage = usage % dict(prog=self._prog)

        # if no optionals or positionals are available, usage is just prog
        elif usage is None and not actions:
            usage = "%(prog)s" % dict(prog=self._prog)
        elif usage is None:
            prog = "%(prog)s" % dict(prog=self._prog)
            # build full usage string
            action_usage = self._format_actions_usage(actions, groups)  # NEW
            usage = " ".join([s for s in [prog, action_usage] if s])
            # omit the long line wrapping code
        # prefix with 'usage:'
        return "%s%s\n\n" % (prefix, usage)


activity_distributions = {"negpowerlaw": negpowerlaw}
ec_types = {
    "SocialNetwork": SocialNetwork,
    "SampleChamber": SampleChamber,
    "ConnChamber": ConnChamber,
    "OpenChamber": OpenChamber,
    "ContrastChamber": ContrastChamber,
}
# defaults
N = 1000
m = 10  # number of other agents to interact with
alpha = 2.0  # controversialness of issue (sigmoidal shape)
K = 3.0  # social interaction strength
gamma = 2.1  # power law distribution param
epsilon = 1e-2  # minimum activity level with another agent
dist_args = [gamma, epsilon]
dist_args_str = [str(d) for d in dist_args]
beta = 2  # power law decay of connection probability
r = 0.5  # probability of mutual interaction

D = 0.01  # noise in SocialNetwork dynamics

dt = 0.01
t_dur = 0.5

# noinspection PyTypeChecker
parser = argparse.ArgumentParser(
    formatter_class=Formatter, description="Run a network of agents."
)
parser.add_argument(
    "N",
    type=int,
    default=N,
    help="Number of agents in the SocialNetwork.",
)
parser.add_argument(
    "m",
    type=int,
    default=m,
    help="Number of other agents to interact with.",
)
parser.add_argument(
    "alpha",
    type=float,
    default=alpha,
    help="Controversialness of issue.",
)
parser.add_argument("K", type=float, default=K, help="Social interaction strength.")
parser.add_argument(
    "-D",
    "--noise",
    nargs="*",
    metavar="D",
    type=float,
    default=[0.0],
    help="Nudge strength.",
)
parser.add_argument(
    "-n",
    "--sample_size",
    nargs="*",
    metavar="n",
    type=float,
    default=[1.0],
    help="Sample size.",
)
parser.add_argument(
    "-cls",
    "--network-type",
    type=str,
    default="SocialNetwork",
    choices=ec_types.keys(),
    help="Type of network dynamics.",
)
parser.add_argument(
    "-a",
    "--activity",
    metavar="ACTIVITY ARG",
    type=str,
    nargs="+",
    default=["negpowerlaw"] + dist_args_str,
    help=f"Distribution to sample activities. If multiple values the 2nd onwards are used as "
    f"distribution parameters.\nDefault: negpowerlaw {' '.join(dist_args_str)}",
)
parser.add_argument(
    "-beta",
    "--connection_decay",
    metavar="BETA",
    dest="beta",
    type=float,
    default=beta,
    help=f"Power law decay of connection probability (0 means uniform distribution of connecting with other "
    f"agents).\nDefault: {beta}",
)
parser.add_argument(
    "-r",
    "--p_mutual_connection",
    metavar="P",
    dest="r",
    type=float,
    default=r,
    help=f"Probability of a mutual connection.\nDefault: {r}",
)
sim_params = parser.add_argument_group("simulation arguments")
sim_params.add_argument(
    "-dt",
    "--time_step",
    metavar="DT",
    dest="dt",
    type=float,
    default=dt,
    help=f"Step size for numerical integrator.\nDefault: {dt}",
)
sim_params.add_argument(
    "-T",
    "--time_end",
    metavar="T",
    dest="T",
    type=float,
    default=t_dur,
    help=f"Simulation length. Default: {t_dur}",
)

sim_params.add_argument(
    "--method",
    type=str,
    default=None,
    choices=list(ODE_INTEGRATORS) + list(SDE_INTEGRATORS) + list(SCIPY_METHODS),
    help="Numerical method of integration. If noise is added to the system, the dynamics become sets of stochastic "
    "differential equations (SDE) instead of ordinary differential equations (ODE)."
    "\nDefault ODE solver: Runge-Kutta order 5(4)"
    "\nDefault SDE solver: Euler-Maruyama",
)

display_params = parser.add_argument_group("display arguments")

display_params.add_argument(
    "--plot",
    dest="plot",
    type=str,
    nargs="?",
    const=True,
    default=False,
    choices=[True, False, "summary", "all"],
    help="Display the results.\nDefault: False",
)

display_params.add_argument(
    "--save",
    type=str,
    nargs="?",
    const=True,
    default=False,
    choices=[True, False, "png", "eps", "pdf"],
    help="Save the figure results.\nDefault: False",
)

display_params.add_argument(
    "-v",
    "--verbose",
    action="store_const",
    const=True,
    default=False,
    help="Verbose terminal output.",
)

args = parser.parse_args()

level = logging.DEBUG if args.verbose else logging.INFO

logging.basicConfig(level=level)
logging.getLogger().setLevel(level)

logger = logging.getLogger("opdynamics")

activity_distribution = activity_distributions[args.activity[0]]
for i in range(1, len(args.activity)):
    arg = float(args.activity[i])
    if i >= len(dist_args) + 1:
        dist_args.append(arg)
    else:
        # overwrite defaults
        dist_args[i - 1] = arg

kwargs = dict(
    N=args.N,
    m=args.m,
    K=args.K,
    alpha=args.alpha,
    beta=args.beta,
    activity_distribution=activity_distribution,
    epsilon=dist_args[1],
    gamma=dist_args[0],
    dt=args.dt,
    T=args.T,
    r=args.r,
    D=args.D,
    sample_size=args.n,
    method=args.method,
    plot_opinion=args.plot,
    cache="all",
)

ec_type = ec_types[args.cls]

# Run simulation
D_range = args.D
range_parameters = {"D": {"range": D_range}, "title": "D"}
_sns = run_product(
    range_parameters,
    cls=ec_type,
    cache="all",
    cache_sim=False,
    parallel=True,
    **kwargs,
)

if args.plot:
    import matplotlib.pyplot as plt

    for sn in _sns:
        show_simulation_results(sn, args.plot)
        if args.save:
            figs = plt.get_fignums()
            # default to pdf
            fmt = args.save if type(args.save) is str else "pdf"
            # get title from passed arguments (ignore None and unwanted keys)
            args_kv = sorted(args.__dict__.items())
            title = ",".join(
                [
                    f"{k}={v}"
                    for k, v in args_kv
                    if v is not None and k not in ["save", "verbose", "plot"]
                ]
            )
            for fig_num in figs:
                fig: plt.Figure = plt.figure(fig_num)
                # add title directly to figure (can be removed in vector-editing software for non-rasterized formats)
                fig.suptitle(title, fontsize="xx-small")
                # try find an ax title to use to name the file
                name = "summary" if args.plot == "summary" else None
                for ax in fig.axes:
                    if fmt == "pdf":
                        # speed up saving and conserve space
                        ax.set_rasterized(True)
                    if name is None or name == "":
                        name = ax.get_title()
                if name is None or name == "":
                    name = fig_num
                else:
                    name = "_".join(name.lower().split())

                filename = f"{sn.name}_{title}_{name}.{fmt}"
                logger.info(f"saving '{filename}'")
                try:
                    fig.savefig(filename, dpi=600)
                except IOError:
                    # probably file name too long, remove the long title part
                    fig.savefig(f"{sn.name}_{name}.{fmt}", dpi=600)
    logger.info("showing plots")
    plt.show()
