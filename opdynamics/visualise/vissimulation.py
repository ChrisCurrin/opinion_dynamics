from opdynamics.utils.constants import (
    POST_RDN_COLOR,
    POST_RECOVERY_COLOR,
    PRE_RDN_COLOR,
)
import opdynamics.visualise.compat

import logging

import numpy as np

from opdynamics.socialnetworks import SocialNetwork, NoisySocialNetwork

logger = logging.getLogger("vis simulation")


def show_simulation_results(_ec: SocialNetwork, plot_opinion: str = None):
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    from opdynamics.visualise.vissocialnetwork import VisSocialNetwork
    import seaborn as sns

    logger.debug("plotting")
    _vis = VisSocialNetwork(_ec)
    if plot_opinion == "summary":
        fig, ax = _vis.show_summary(single_fig=True)
        fig.subplots_adjust(wspace=0.5, hspace=0.3, top=0.95, right=0.9)
        sns.despine()
    elif plot_opinion == "all":
        _vis.show_summary(single_fig=False)
        sns.despine()
    else:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(
            1, 2, sharey="col", gridspec_kw={"width_ratios": [1, 0.1]}
        )
        _vis.show_opinions(True, ax=axs[0])
        _vis.show_opinions_distribution(vertical=True, ax=axs[-1])
        axs[-1].set_ylabel("")
        axs[-1].set_yticklabels([])
        fig.subplots_adjust(hspace=0.1)


def show_simulation_range(sn_arr, fig_ax=None):
    from opdynamics.visualise.vissocialnetwork import VisSocialNetwork
    import matplotlib.pyplot as plt
    import seaborn as sns

    cs = sns.color_palette("husl", n_colors=len(sn_arr))
    if fig_ax is not None and type(fig_ax) is tuple:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(
            nrows=len(sn_arr), ncols=1, sharex="all", sharey="col", figsize=(8, 11)
        )
    logger.debug(f"ax.shape = {ax.shape}")
    for i, (nsn, _ax) in enumerate(zip(sn_arr, ax)):
        vis = VisSocialNetwork(nsn)
        # 0 is first column, 1 is 2nd column
        kwargs = {"ax": _ax, "color": cs[i]}
        if i > 0:
            kwargs["title"] = False
        vis.show_opinions_distribution(**kwargs)
        if i != len(sn_arr) - 1:
            _ax.set_xlabel("")
        _ax.set_ylabel(
            vis.sn.name,
            color=cs[i],
            fontsize="x-large",
            rotation=0,
            va="top",
            ha="right",
        )

        _ax.grid(True, axis="x")
    sns.despine()


def show_periodic_noise(
    nsn: NoisySocialNetwork, noise_start, noise_length, recovery, interval, num, D
):
    logger.debug("plotting periodic noise")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec
    from opdynamics.visualise import VisSocialNetwork
    from opdynamics.utils.plot_utils import get_time_point_idx

    # calculate optimal bin edges from opinions distribution at noise start + noise_length
    hist, bin_edges = np.histogram(
        nsn.result.y[
            :, get_time_point_idx(nsn.result.t, float(noise_start + noise_length))
        ],
        bins="auto",
    )

    vis = VisSocialNetwork(nsn)
    # create figure and axes
    fig = plt.figure()
    gs = gridspec.GridSpec(
        nrows=2, ncols=3, figure=fig, wspace=0.3, hspace=0.8, height_ratios=(1, 2)
    )
    ax_time = fig.add_subplot(gs[0, :])
    ax_start = fig.add_subplot(gs[-1, 0])
    ax_noise = fig.add_subplot(gs[-1, 1], sharey=ax_start)
    ax_recovery = fig.add_subplot(gs[-1, 2], sharey=ax_start)

    _colors = [PRE_RDN_COLOR, POST_RDN_COLOR, POST_RECOVERY_COLOR]
    # plot graphs
    vis.show_opinions(ax=ax_time, color_code="line", subsample=5, title=False)
    vis.show_opinions_distribution(
        ax=ax_start,
        t=noise_start,
        title=f"t = {noise_start}",
        color=PRE_RDN_COLOR,
        bins=bin_edges,
    )
    vis.show_opinions_distribution(
        ax=ax_noise,
        t=noise_start + noise_length,
        title=f"t={noise_start + noise_length}",
        color=POST_RDN_COLOR,
        bins=bin_edges,
    )
    vis.show_opinions_distribution(
        ax=ax_recovery,
        t=-1,
        title=f"t={noise_start + noise_length + recovery}",
        color=POST_RECOVERY_COLOR,
        bins=bin_edges,
    )
    # adjust view limits
    from scipy import stats

    x_data, y_data = nsn.result.t, nsn.result.y
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
        clip_on=False,
    )
    # noise on/off
    noiseless_time = interval * (num - 1)
    block_time = (noise_length - noiseless_time) / num
    block_times_s = [noise_start + block_time * i + interval * i for i in range(num)]
    block_times_e = [
        noise_start + block_time * (i + 1) + interval * i for i in range(num)
    ]
    ax_time.hlines(
        y=[lim[1]] * num,
        xmin=block_times_s,
        xmax=block_times_e,
        lw=5,
        color="k",
        clip_on=False,
    )
    # value of noise
    ax_time.annotate(f"D = {D}", xy=(noise_start, lim[1]), ha="left", va="bottom")
    # recovery annotation
    # ax_time.annotate(
    #     f"D = 0",
    #     xy=(noise_start + noise_length, lim[1]),
    #     ha="left",
    #     va="bottom",
    # )
    sns.despine()
    ax_noise.set_ylabel("")
    ax_recovery.set_ylabel("")
    return fig, gs
