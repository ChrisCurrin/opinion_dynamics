""""""
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from opdynamics.networks import EchoChamber
from opdynamics.utils.constants import *
from opdynamics.utils.decorators import optional_fig_ax
from opdynamics.utils.plot_utils import use_self_args
from opdynamics.utils.plot_utils import colorbar_inset, colorline
from opdynamics.visualise.dense import show_activity_vs_opinion, show_matrix

logger = logging.getLogger("visualise")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class VisEchoChamber(object):
    """Class used to visualise attached EchoChamber object.

    Visualisations
    =====
    - ``show_activities``
    - ``show_opinions``
    - ``show_opinions_snapshot``
    - ``show_agent_opinions``
    - ``show_adjacency_matrix``
    - ``show_nearest_neighbour``
    - ``show_summary``


    Private methods
    ----
     - ``_get_equal_opinion_limits``

    Examples
    =====

    Normal usage
    -----

    .. code-block :: python

        vis = VisEchoChamber(ec)
        vis.show_opinions()

    One-time use
    -----

    .. code-block :: python

        VisEchoChamber(ec).show_opinions()

    """

    def __init__(self, echochamber: EchoChamber):
        self.ec = echochamber

    show_activity_vs_opinion = use_self_args(
        show_activity_vs_opinion, ["ec.opinions", "ec.activities"]
    )

    def show_connection_probabilities(
        self, *args, title="Connection probabilities", **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot connection probability matrix.

        :keyword map: How to plot the matrix.
                * 'clustermap' - `sns.clustermap`
                * 'heatmap' - `sns.heatmap`
                * 'mesh' - `matplotlib.pcolormesh`
                * callable - calls the function using the (potentially generated) ax, norm, and keywords.
        :keyword sort: Sort the matrix by most interactions.
        :keyword norm: The scale of the plot (normal, log, etc.).
        :keyword cmap: Colormap to use.
        :keyword ax: Axes to use for plot. Created if none passed.
        :keyword fig: Figure to use for colorbar. Created if none passed.
        :keyword title: Include title in the figure.

        :keyword cbar_ax: Axes to use for plotting the colorbar.

        :return: (Figure, Axes) used.

        """
        # noinspection PyProtectedMember
        p_conn = self.ec.adj_mat._p_conn
        if p_conn is None:
            p_conn = self.ec.adj_mat.conn_method(self.ec, **self.ec.adj_mat.conn_kwargs)
        return show_matrix(
            p_conn,
            "$P_{ij}$",
            *args,
            # min value must be > 0 when LogNorm is used
            vmin=np.min(p_conn[p_conn > 0]),
            vmax=1,
            title=title,
            **kwargs,
        )

    def show_adjacency_matrix(
        self, *args, title="Cumulative adjacency matrix", **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot adjacency matrix.

        If matrix `ec.adj_mat` is 3D (first dimension is time), the sum of the interactions is computed over time.
        The total adjacency matrix

        > Note adj_mat is indexed ji but represents Aij (if there is input from agent j to agent i)

        :keyword map: How to plot the matrix.
                * 'clustermap' - `sns.clustermap`
                * 'heatmap' - `sns.heatmap`
                * 'mesh' - `matplotlib.pcolormesh`
                * callable - calls the function using the (potentially generated) ax, norm, and keywords.
        :keyword sort: Sort the matrix by most interactions.
        :keyword norm: The scale of the plot (normal, log, etc.).
        :keyword cmap: Colormap to use.
        :keyword ax: Axes to use for plot. Created if none passed.
        :keyword fig: Figure to use for colorbar. Created if none passed.
        :keyword title: Include title in the figure.

        :keyword cbar_ax: Axes to use for plotting the colorbar.

        :return: (Figure, Axes) used.

        """
        return show_matrix(
            self.ec.adj_mat.accumulator,
            "Number of interactions",
            *args,
            vmin=1,
            title=title,
            **kwargs,
        )

    # alias
    show_interactions = show_adjacency_matrix

    @optional_fig_ax
    def show_activities(
        self, ax: Axes = None, fig: Figure = None, **kwargs
    ) -> (Figure, Axes):
        kwargs.setdefault("color", "Green")
        sns.histplot(self.ec.activities, axlabel="activity", ax=ax, **kwargs)
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
        title: str = "Opinion dynamics",
        **kwargs,
    ) -> (Figure, Axes):
        """
        Display the evolution of opinions over time.

        :param color_code: Whether to color by valence (True) or by agent (False).
            By default, a scatter plot is used, but a line plot with precise coloring can be specified with 'line'.
        :param subsample: The number of agents to plot.
            A subsample of 1 means every agent is plotted. 10 is every 10th agent, etc.
        :param ax: Axes to use for plot. Created if none passed.
        :param fig: Figure to use for colorbar. Created if none passed.
        :param title: Include title in the figure.

        :keyword cmap: Colormap to use (color_code = True or 'line')
        :keyword vmin: minimum value to color from cmap. Lower values will be colored the same.
        :keyword vmax: maximum value to color from cmap. Higher values will be colored the same.

        :return: (Figure, Axes) used.
        """
        cmap = kwargs.pop("cmap", OPINIONS_CMAP)
        vmin = kwargs.pop("vmin", np.min(self.ec.result.y))
        vmax = kwargs.pop("vmax", np.max(self.ec.result.y))
        sm = ScalarMappable(norm=TwoSlopeNorm(0, vmin, vmax), cmap=OPINIONS_CMAP)
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
                    self.ec.result.t,
                    n,
                    ls=ls,
                    mec=mec,
                    lw=lw,
                    ax=ax,
                    **kwargs,
                )
        ax.set_xlim(0, self.ec.result.t[-1])
        ax.set_xlabel(TIME_SYMBOL)
        ax.set_ylabel(OPINION_AGENT_TIME)
        ax.set_ylim(*self._get_equal_opinion_limits())
        if title:
            ax.set_title(title)
        return fig, ax

    @optional_fig_ax
    def show_opinions_snapshot(
        self,
        t=-1,
        ax: Axes = None,
        fig: Figure = None,
        title: str = "Opinions distribution",
        **kwargs,
    ) -> (Figure, Axes):
        idx = np.argmin(np.abs(t - self.ec.result.t)) if isinstance(t, float) else t
        bins = kwargs.pop("bins", "auto")
        kwargs.setdefault("color", "Purple")
        kwargs.setdefault("kde", True)
        sns.histplot(self.ec.result.y[:, idx], bins=bins, ax=ax, **kwargs)

        vertical = kwargs.get("vertical", False)
        if vertical:
            ax.set_ylabel(OPINION_SYMBOL)
            ax.set_xlabel(f"$P({OPINION_SYMBOL})$")
            ax.set_ylim(*self._get_equal_opinion_limits())
        else:
            ax.set_xlabel(OPINION_SYMBOL)
            ax.set_ylabel(f"$P({OPINION_SYMBOL})$")
            ax.set_xlim(*self._get_equal_opinion_limits())
        if title:
            ax.set_title(title)
        return fig, ax

    @optional_fig_ax
    def show_agent_opinions(
        self,
        t=-1,
        direction=True,
        sort=False,
        ax: Axes = None,
        fig: Figure = None,
        colorbar: bool = True,
        title: str = "Agent opinions",
    ) -> (Figure, Axes):
        idx = np.argmin(np.abs(t - self.ec.result.t)) if isinstance(t, float) else t
        opinions = self.ec.result.y[:, idx]
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
        sm = ScalarMappable(norm=Normalize(*v), cmap=OPINIONS_CMAP)
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
            ax.hlines(
                y=min_idx,
                xmin=v[0],
                xmax=v[1],
                ls="--",
                color="k",
                alpha=0.5,
                lw=1,
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
            cbar = colorbar_inset(sm, "outer bottom", size="5%", pad=0.01, ax=ax)
            sns.despine(ax=ax, bottom=True)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            cbar.set_label(OPINION_SYMBOL)

        ax.set_ylim(0, self.ec.N)
        ax.set_xlim(*v)
        if not colorbar:
            # xlabel not part of colorbar
            ax.set_xlabel(OPINION_SYMBOL)
        ax.set_ylabel("Agent $i$")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if title:
            ax.set_title(title)
        return fig, ax

    def _get_equal_opinion_limits(self):
        if self.ec.result is None:
            opinions = self.ec.opinions
        else:
            opinions = self.ec.result.y
        v = np.max(np.abs(opinions))
        return -v, v

    def show_nearest_neighbour(
        self, bw=0.5, t=-1, title=True, **kwargs
    ) -> sns.JointGrid:
        nn = self.ec.get_nearest_neighbours(t)
        kwargs.setdefault("color", "Purple")
        marginal_kws = kwargs.pop("marginal_kws", dict())
        marginal_kws.update(bw=bw)
        g = sns.jointplot(
            self.ec.opinions, nn, kind="kde", bw=bw, marginal_kws=marginal_kws, **kwargs
        )
        g.ax_joint.set_xlabel(OPINION_SYMBOL)
        g.ax_joint.set_ylabel(MEAN_NEAREST_NEIGHBOUR)
        if title:
            g.fig.suptitle("Neighbour's opinions", va="bottom")
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
        _, ax[2, 1], cbar = self.show_activity_vs_opinion(ax=ax[2, 1])
        # `show_agent_opinions` already calculates optimal limits
        xlim = ax[1, 1].get_xlim()
        ax[0, 1].set_xlim(*xlim)
        ax[2, 1].set_xlim(*xlim)
        return fig, ax
