""""""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable

from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import interpn
from seaborn.matrix import ClusterGrid

from opdynamics.echochamber import EchoChamber
from opdynamics.utils.constants import *
from opdynamics.utils.decorators import optional_fig_ax
from opdynamics.utils.plot_utils import colorbar_inset

logger = logging.getLogger("visualise")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class VisEchoChamber(object):
    def __init__(self, echochamber: EchoChamber):
        self.ec = echochamber

    @optional_fig_ax
    def show_activities(self, ax: Axes = None, fig: Figure = None) -> (Figure, Axes):
        sns.distplot(self.ec.activities, kde=False, axlabel="activity", ax=ax)
        ax.set(title="Activity distribution", ylabel="count")
        return fig, ax

    @optional_fig_ax
    def show_opinions(
        self, color_code=True, ax: Axes = None, fig: Figure = None
    ) -> (Figure, Axes):
        b = colors.to_rgba("blue")
        r = colors.to_rgba("red")
        if color_code == "line" or color_code == "lines":
            # using the colorline method allows colors to be dependent on a value, in this case, opinion,
            # but takes much longer to display
            for n in self.ec.result.y:
                z = np.ndarray(shape=[len(n), 4])
                mask = n >= 0
                z[mask] = b
                z[~mask] = r
                colorline(self.ec.result.t, n, z, lw=0.1, ax=ax)
        elif color_code:
            for n in self.ec.result.y:
                z = np.ndarray(shape=[len(n), 4])
                mask = n >= 0
                z[mask] = b
                z[~mask] = r
                ax.scatter(self.ec.result.t, n, c=z, s=0.1)
        else:
            sns.set_palette(sns.color_palette("Set1", n_colors=len(self.ec.result.y)))
            for n in self.ec.result.y:
                sns.lineplot(
                    self.ec.result.t, n, linestyle="-", mec="None", lw=0.1, ax=ax,
                )
        ax.set_xlim(0, self.ec.result.t[-1])
        ax.set_xlabel(TIME_SYMBOL)
        ax.set_ylabel(OPINION_AGENT_TIME)
        ax.set_title("Opinion dynamics")
        return fig, ax

    @optional_fig_ax
    def show_opinions_snapshot(
        self, ax: Axes = None, fig: Figure = None, **kwargs
    ) -> (Figure, Axes):
        sns.distplot(self.ec.opinions, ax=ax, **kwargs)
        ax.set_xlabel(OPINION_SYMBOL)
        ax.set_ylabel(f"$P({OPINION_SYMBOL})$")
        ax.set_title("Opinions distribution")
        return fig, ax

    @optional_fig_ax
    def show_agent_opinions(
        self,
        direction=True,
        sort=False,
        ax: Axes = None,
        fig: Figure = None,
        colorbar=True,
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

        # center the colormap on 0 by equally spacing vmin and vmax
        v = np.max(np.abs(opinions))  # largest value
        sm = ScalarMappable(norm=Normalize(-v, v), cmap="seismic_r")
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
        ax.set_xlim(-v, v)
        if not colorbar:
            # xlabel not part of colorbar
            ax.set_xlabel(OPINION_SYMBOL)
        ax.set_ylabel("Agent $i$")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_title("Agent opinions")
        return fig, ax

    @optional_fig_ax
    def show_activity_vs_opinions(
        self,
        bins: int = 20,
        norm: Normalize = LogNorm(),
        ax: Axes = None,
        fig: Figure = None,
        **kwargs,
    ) -> (Figure, Axes):
        """
        Density scatter plot colored by 2d histogram

        https://stackoverflow.com/a/53865762

        :param bins: Number of bins to group opinions for determining density.
        :param norm: The scale of the density plot (normal, log, etc.)
        :param ax: Axes to use for plot. Created if none passed.
        :param fig: Figure to use for colorbar. Created if none passed.
        :return: (Figure, Axes) used.
        """
        # get density based on bin size
        data, x_e, y_e = np.histogram2d(
            self.ec.opinions, self.ec.activities, bins=bins, density=True
        )
        # interpolate so shape is same as x and y
        z = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            data,
            np.vstack([self.ec.opinions, self.ec.activities]).T,
            method="splinef2d",
            bounds_error=False,
        )

        # To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last

        idx = z.argsort()
        x, y, z = self.ec.opinions[idx], self.ec.activities[idx], z[idx]
        marker_size = kwargs.pop("s", 1)
        points = ax.scatter(x, y, s=marker_size, c=z, norm=norm, **kwargs)

        # create colorbar axes without stealing from main ax
        colorbar_inset(points, "outer right", size="5%", ax=ax, label=P_A_X)

        # fig.colorbar(points, cax=cax, label=P_A_X)

        ax.set_xlabel(OPINION_SYMBOL)
        ax.set_ylabel(ACTIVITY_SYMBOL)
        ax.set_title("Density of activity and opinions")
        return fig, ax

    @optional_fig_ax
    def show_adjacency_matrix(
        self,
        map="clustermap",
        sort=False,
        norm=LogNorm(),
        ax: Axes = None,
        fig: Figure = None,
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

        # default label for colorbar
        cbar_kws = {"label": "Number of interactions", **kwargs.pop("cbar_kws", {})}

        if isinstance(norm, LogNorm) and getattr(kwargs, "vmin", -1) <= 0:
            kwargs["vmin"] = 1

        if sort:
            total_interactions = total_interactions.sort_values(
                by=list(total_interactions.index), axis="index"
            )
            total_interactions = total_interactions.sort_values(
                by=list(total_interactions.columns), axis="columns"
            )

        if map == "clustermap":
            fig = sns.clustermap(
                total_interactions, norm=norm, cbar_kws=cbar_kws, **kwargs
            )
            ax = fig.ax_heatmap
        elif map == "heatmap":
            sns.heatmap(
                total_interactions, norm=norm, ax=ax, cbar_kws=cbar_kws, **kwargs
            )
            ax.invert_yaxis()
        elif map == "mesh":
            if sort:
                logger.warning(
                    "'mesh' loses agent index information when sorting adjacency matrix"
                )
            mesh = ax.pcolormesh(total_interactions, norm=norm, **kwargs)
            ax.set_xlim(0, self.ec.N)
            ax.set_ylim(0, self.ec.N)
            cax = cbar_kws.pop("cbar_ax", None) or cbar_kws.pop("cax", None)
            if cax is None:
                colorbar_inset(
                    ScalarMappable(norm=norm, cmap=getattr(kwargs, "cmap", None)),
                    "outer right",
                    size="5%",
                    ax=ax,
                    **cbar_kws,
                )
            else:
                # steal space from ax
                fig.colorbar(
                    mesh, cmap=getattr(kwargs, "cmap", None), ax=ax, **cbar_kws,
                )
        elif isinstance(map, Callable):
            map(total_interactions, ax=ax, norm=norm, **kwargs)
        else:
            raise NotImplementedError(
                f"Method {map} not implemented. Try one of 'clustermap' 'heatmap' 'mesh' or a "
                f"function."
            )
        ax.set_xlabel("Agent $i$")
        ax.set_ylabel("Agent $j$")
        ax.set_title("Cumulative adjacency matrix")
        return fig, ax

    def show_nearest_neighbour(self, **kwargs,) -> sns.JointGrid:
        nn = self.ec.get_nearest_neighbours()
        g = sns.jointplot(self.ec.opinions, nn, kind="kde", **kwargs)
        g.ax_joint.set_xlabel(OPINION_SYMBOL)
        g.ax_joint.set_ylabel(MEAN_NEAREST_NEIGHBOUR)
        g.fig.suptitle("Neighbour's opinions")
        return g

    def show_summary(self, single_fig=True, fig_kwargs=None):
        nrows = 3
        ncols = 2
        if single_fig:
            if fig_kwargs is None:
                fig_kwargs = {
                    "figsize": (8, 8),
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


# Methods which may be generally useful.


def colorline(
    x,
    y,
    z=None,
    cmap="seismic",
    norm=Normalize(0.0, 1.0),
    linewidth=1.0,
    ax=None,
    **kwargs,
):
    """
    Plot a colored line with coordinates x and y

    Optionally specify colors in the array z

    Optionally specify a colormap, a norm function and a line width

    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb

    http://matplotlib.org/examples/pylab_examples/multicolored_line.html

    """

    import matplotlib.collections as mcoll

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = _make_segments(x, y)
    lc = mcoll.LineCollection(
        segments, colors=z, cmap=cmap, norm=norm, linewidth=linewidth, **kwargs
    )

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc


@optional_fig_ax
def show_K_alpha_phase(df: pd.DataFrame, ax=None, fig=None):
    im = ax.pcolormesh(df.columns, df.index, np.abs(df.values))
    fig.colorbar(im, label=ABS_MEAN_FINAL_OPINION)
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$K$")
    return fig, ax


# Methods used within this file only


def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
