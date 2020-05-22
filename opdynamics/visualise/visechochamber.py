""""""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable, Tuple

from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from seaborn.matrix import ClusterGrid

from opdynamics.dynamics.echochamber import EchoChamber
from opdynamics.utils.constants import *
from opdynamics.utils.decorators import optional_fig_ax
from opdynamics.utils.plot_utils import use_self_args
from opdynamics.utils.plot_utils import colorbar_inset, colorline
from opdynamics.visualise.dense import show_activity_vs_opinion

logger = logging.getLogger("visualise")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class VisEchoChamber(object):
    def __init__(self, echochamber: EchoChamber):
        self.ec = echochamber

    show_activity_vs_opinion = use_self_args(
        show_activity_vs_opinion, ["ec.opinions", "ec.activities"]
    )

    @optional_fig_ax
    def show_activities(self, ax: Axes = None, fig: Figure = None) -> (Figure, Axes):
        sns.distplot(self.ec.activities, kde=False, axlabel="activity", ax=ax)
        ax.set(
            title="Activity distribution",
            ylabel="count",
            xlim=(
                np.min(np.append(self.ec.activities, 0)),
                np.max(np.append(self.ec.activities, 1)),
            ),
        )
        return fig, ax

    @optional_fig_ax
    def show_opinions(
        self,
        color_code=True,
        subsample: int = 1,
        ax: Axes = None,
        fig: Figure = None,
        title: bool = True,
        **kwargs,
    ) -> (Figure, Axes):
        """

        :param color_code: Whether to color by valence (True) or by agent (False).
            By default, a scatter plot is used, but a line plot with precise coloring can be specified with 'line'.
        :param subsample: The number of agents to plot.
            A subsample of 1 means every agent is plotted. 10 is every 10th agent, etc.
        :param ax: Axes to use for plot. Created if none passed.
        :param fig: Figure to use for colorbar. Created if none passed.
        :param title: Include title in the figure.


        :return: (Figure, Axes) used.
        """
        sm = ScalarMappable(
            norm=TwoSlopeNorm(0, np.min(self.ec.result.y), np.max(self.ec.result.y)),
            cmap=OPINIONS_CMAP,
        )
        if color_code == "line" or color_code == "lines":
            # using the colorline method allows colors to be dependent on a value, in this case, opinion,
            # but takes much longer to display
            for n in self.ec.result.y[::subsample]:
                c = sm.to_rgba(n)
                lw = kwargs.pop("lw", 0.1)
                colorline(self.ec.result.t, n, c, lw=lw, ax=ax, **kwargs)
        elif color_code:
            for n in self.ec.result.y[::subsample]:
                c = sm.to_rgba(n)
                s = kwargs.pop("s", 0.1)
                ax.scatter(self.ec.result.t, n, c=c, s=s, **kwargs)
        else:
            lw = kwargs.pop("lw", 0.1)
            ls = kwargs.pop("ls", "-")
            mec = kwargs.pop("mec", "None")
            sns.set_palette(sns.color_palette("Set1", n_colors=len(self.ec.result.y)))
            for n in self.ec.result.y[::subsample]:
                sns.lineplot(
                    self.ec.result.t, n, ls=ls, mec=mec, lw=lw, ax=ax,
                )
        ax.set_xlim(0, self.ec.result.t[-1])
        ax.set_xlabel(TIME_SYMBOL)
        ax.set_ylabel(OPINION_AGENT_TIME)
        ax.set_ylim(*self._get_equal_opinion_limits())
        if title:
            if type(title) is not bool:
                ax.set_title(title)
            else:
                ax.set_title("Opinion dynamics")
        return fig, ax

    @optional_fig_ax
    def show_opinions_snapshot(
        self, ax: Axes = None, fig: Figure = None, title: bool = True, t=-1, **kwargs
    ) -> (Figure, Axes):
        idx = np.argmin(np.abs(t - self.ec.result.t)) if isinstance(t, float) else t
        bins = kwargs.pop("bins", self.ec.N // 5)
        sns.distplot(self.ec.result.y[:, idx], bins=bins, ax=ax, **kwargs)
        ax.set_xlabel(OPINION_SYMBOL)
        ax.set_ylabel(f"$P({OPINION_SYMBOL})$")
        ax.set_xlim(*self._get_equal_opinion_limits())
        if title:
            ax.set_title(title or "Opinions distribution")
        return fig, ax

    @optional_fig_ax
    def show_agent_opinions(
        self,
        direction=True,
        sort=False,
        ax: Axes = None,
        fig: Figure = None,
        colorbar: bool = True,
        title: bool = True,
    ) -> (Figure, Axes):
        opinions = self.ec.opinions
        agents = np.arange(self.ec.N)
        if not direction:
            # only magnitude
            opinions = np.abs(opinions)

        if sort:
            logger.warning(
                "sorting opinions for `show_agent_opinions` means agent indices are jumbled"
            )
            # sort by opinion
            ind = np.argsort(opinions)
            opinions = opinions[ind]
            # agents = agents[ind]

        v = self._get_equal_opinion_limits()
        sm = ScalarMappable(
            norm=TwoSlopeNorm(0, np.min(opinions), np.max(opinions)), cmap=OPINIONS_CMAP
        )
        color = sm.to_rgba(opinions)

        ax.barh(
            agents,
            opinions,
            color=color,
            edgecolor="None",
            linewidth=0,  # remove bar borders
            height=1,  # per agent
        )
        ax.axvline(x=0, ls="-", color="k", alpha=0.5, lw=1)
        if sort:
            min_idx = np.argmin(np.abs(opinions))
            ax.axhline(
                y=min_idx, ls="--", color="k", alpha=0.5, lw=1,
            )
            ax.annotate(
                f"{min_idx}",
                xy=(np.min(opinions), min_idx),
                fontsize="small",
                color="k",
                alpha=0.5,
                va="bottom",
                ha="left",
            )
        if colorbar:
            # create colorbar axes without stealing from main ax
            cbar = colorbar_inset(sm, "outer bottom", size="5%", ax=ax,)
            ax.set_xticklabels([])
            cbar.set_label(OPINION_SYMBOL)

        ax.set_ylim(0, self.ec.N)
        if not colorbar:
            # xlabel not part of colorbar
            ax.set_xlabel(OPINION_SYMBOL)
        ax.set_ylabel("Agent $i$")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if title:
            ax.set_title("Agent opinions")
        return fig, ax

    def show_activity_vs_opinions(self, *args, **kwargs,) -> (Figure, Axes):
        return show_activity_vs_opinion(
            self.ec.opinions, self.ec.activities, *args, **kwargs
        )

    def _get_equal_opinion_limits(self):
        if self.ec.result is None:
            opinions = self.ec.opinions
        else:
            opinions = self.ec.result.y
        v = np.max(np.abs(opinions))
        return -v, v

    @optional_fig_ax
    def show_adjacency_matrix(
        self,
        map="clustermap",
        sort=False,
        norm=LogNorm(),
        cmap=INTERACTIONS_CMAP,
        ax: Axes = None,
        fig: Figure = None,
        title: bool = True,
        **kwargs,
    ) -> (Figure or ClusterGrid, Axes):
        """
        Plot adjacency matrix.

        If matrix `ec.adj_mat` is 3D (first dimension is time), the sum of the interactions is computed over time.
        The total adjacency matrix

        > Note adj_mat is indexed ji but represents Aij (if there is input from agent j to agent i)

        :param map: How to plot the matrix.
            * 'clustermap' - `sns.clustermap`
            * 'heatmap' - `sns.heatmap`
            * 'mesh' - `matplotlib.pcolormesh`
            * callable - calls the function using the (potentially generated) ax, norm, and keywords.
        :param sort: Sort the matrix by most interactions.
        :param norm: The scale of the plot (normal, log, etc.).
        :param ax: Axes to use for plot. Created if none passed.
        :param fig: Figure to use for colorbar. Created if none passed.
        :param title: Include title in the figure.

        :keyword cbar_ax: Axes to use for plotting the colorbar.

        :return: (Figure, Axes) used.
        """

        # cast to DataFrame to keep agent index information when sorting.
        # convert definition of Aij input from j to i to number of interactions by agent i with agent j
        total_interactions = pd.DataFrame(
            self.ec.adj_mat.accumulator,
            columns=pd.Index(np.arange(self.ec.N), name="i"),
            index=pd.Index(np.arange(self.ec.N), name="j"),
        )

        if sort:
            total_interactions = total_interactions.sort_values(
                by=list(total_interactions.index), axis="index"
            ).sort_values(by=list(total_interactions.columns), axis="columns")

        # default label for colorbar
        cbar_kws = {"label": "Number of interactions", **kwargs.pop("cbar_kws", {})}

        if isinstance(norm, LogNorm) and getattr(kwargs, "vmin", -1) <= 0:
            kwargs["vmin"] = 1

        if map == "clustermap":
            if fig:
                plt.close(fig)
            fig = sns.clustermap(
                total_interactions, norm=norm, cmap=cmap, cbar_kws=cbar_kws, **kwargs
            )
            ax = fig.ax_heatmap
        elif map == "heatmap":
            sns.heatmap(
                total_interactions,
                norm=norm,
                cmap=cmap,
                ax=ax,
                cbar_kws=cbar_kws,
                **kwargs,
            )
            ax.invert_yaxis()
        elif map == "mesh":
            if sort:
                logger.warning(
                    "'mesh' loses agent index information when sorting adjacency matrix"
                )
            mesh = ax.pcolormesh(total_interactions, norm=norm, cmap=cmap, **kwargs)
            ax.set_xlim(0, self.ec.N)
            ax.set_ylim(0, self.ec.N)
            cax = cbar_kws.pop("cbar_ax", None) or cbar_kws.pop("cax", None)
            if cax is None:
                colorbar_inset(
                    ScalarMappable(norm=norm, cmap=cmap),
                    "outer right",
                    size="5%",
                    ax=ax,
                    cmap=cmap,
                    **cbar_kws,
                )
            elif isinstance(cax, Axes):
                # using existing cax
                fig.colorbar(
                    mesh, cax=cax, **cbar_kws,
                )
            elif cax:
                # steal space from ax if cax is anything else (i.e. unless None, False, or an Axes)
                fig.colorbar(mesh, ax=ax, **cbar_kws)
        elif isinstance(map, Callable):
            map(total_interactions, ax=ax, norm=norm, **kwargs)
        else:
            raise NotImplementedError(
                f"Method {map} not implemented. Try one of 'clustermap' 'heatmap' 'mesh' or a "
                f"function."
            )
        ax.set_xlabel("Agent $i$")
        ax.set_ylabel("Agent $j$")
        if title:
            if map == "clustermap":
                fig.fig.suptitle("Cumulative adjacency matrix")
            else:
                ax.set_title("Cumulative adjacency matrix")
        return fig, ax

    def show_nearest_neighbour(self, title=True, **kwargs) -> sns.JointGrid:
        nn = self.ec.get_nearest_neighbours()
        bw = kwargs.pop("bw", 0.5)
        marginal_kws = kwargs.pop("marginal_kws", dict())
        marginal_kws.update(bw=bw)
        g = sns.jointplot(
            self.ec.opinions, nn, kind="kde", bw=bw, marginal_kws=marginal_kws, **kwargs
        )
        g.ax_joint.set_xlabel(OPINION_SYMBOL)
        g.ax_joint.set_ylabel(MEAN_NEAREST_NEIGHBOUR)
        if title:
            g.fig.suptitle("Neighbour's opinions")
        return g

    def show_summary(self, single_fig=True, fig_kwargs=None) -> Tuple[Figure, Axes]:
        nrows = 3
        ncols = 2
        if single_fig:
            if fig_kwargs is None:
                fig_kwargs = {
                    "figsize": (8, 11),
                }
            fig, ax = plt.subplots(nrows, ncols, **fig_kwargs)
        else:
            # set all of ax to None so that each method creates its own ax.
            ax = (
                np.tile(np.asarray(None), nrows * ncols)
                .reshape(nrows, ncols)
                .astype(Axes)
            )
            fig = None
            # call other methods that create their own figures
            self.show_nearest_neighbour()
            self.show_adjacency_matrix("clustermap")
        # first column
        _, ax[0, 0] = self.show_opinions(ax=ax[0, 0])
        _, ax[1, 0] = self.show_adjacency_matrix("mesh", sort=True, ax=ax[1, 0])
        _, ax[2, 0] = self.show_activities(ax=ax[2, 0])
        # second column has opinion as x-axis
        _, ax[0, 1] = self.show_opinions_snapshot(ax=ax[0, 1])
        _, ax[1, 1] = self.show_agent_opinions(ax=ax[1, 1], sort=True)
        _, ax[2, 1] = self.show_activity_vs_opinions(ax=ax[2, 1])
        # `show_agent_opinions` already calculates optimal limits
        xlim = ax[1, 1].get_xlim()
        ax[0, 1].set_xlim(*xlim)
        ax[2, 1].set_xlim(*xlim)
        return fig, ax
