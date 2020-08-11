import logging

import numpy as np

from opdynamics.networks import EchoChamber

logger = logging.getLogger("vis simulation")


def show_simulation_results(_ec: EchoChamber, plot_opinion: str):
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
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(
            1, 2, sharey="col", gridspec_kw={"width_ratios": [1, 0.1]}
        )
        _vis.show_opinions(True, ax=axs[0])
        _vis.show_opinions_snapshot(vertical=True, ax=axs[-1])
        axs[-1].set_ylabel("")
        axs[-1].set_yticklabels([])
        fig.subplots_adjust(hspace=0.1)


def show_simulation_range(var_range, nec_arr, fig_ax=None):
    from opdynamics.visualise.visechochamber import VisEchoChamber
    import matplotlib.pyplot as plt
    import seaborn as sns

    cs = sns.color_palette("husl", n_colors=len(var_range))
    if fig_ax is not None and type(fig_ax) is tuple:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots(
            nrows=len(var_range), ncols=1, sharex="all", sharey="col", figsize=(8, 11)
        )
    logger.debug(f"ax.shape = {ax.shape}")
    for i, (nec, _ax) in enumerate(zip(nec_arr, ax)):
        vis = VisEchoChamber(nec)
        # 0 is first column, 1 is 2nd column
        kwargs = {"ax": _ax, "color": cs[i]}
        if i > 0:
            kwargs["title"] = False
        vis.show_opinions_snapshot(**kwargs)
        if i != len(nec_arr) - 1:
            _ax.set_xlabel("")
        _ax.set_ylabel(
            vis.ec.name,
            color=cs[i],
            fontsize="x-large",
            rotation=0,
            va="top",
            ha="right",
        )

        _ax.grid(True, axis="x")
    sns.despine()


def show_periodic_noise(nec, noise_start, noise_length, recovery, interval, num, D):
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
    noiseless_time = interval * (num - 1)
    block_time = (noise_length - noiseless_time) / num
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
    ax_time.annotate(f"noise = {D}", xy=(noise_start, lim[1]), ha="left", va="bottom")
    ax_time.annotate(
        f"noise = 0", xy=(noise_start + noise_length, lim[1]), ha="left", va="bottom",
    )
    sns.despine()
    ax_noise.set_ylabel("")
    ax_recovery.set_ylabel("")
