import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from scipy.interpolate import interpn
from seaborn.matrix import ClusterGrid
from typing import Callable

from opdynamics.dynamics.echochamber import EchoChamber
from opdynamics.utils.decorators import optional_fig_ax
from opdynamics.utils.plot_utils import get_equal_limits, colorbar_inset
from opdynamics.utils.constants import *

logger = logging.getLogger("dense plots")


@optional_fig_ax
def show_K_alpha_phase(df: pd.DataFrame, ax=None, fig=None):
    im = ax.pcolormesh(df.columns, df.index, np.abs(df.values))
    fig.colorbar(im, label=ABS_MEAN_FINAL_OPINION)
    ax.set_xlabel("$\\alpha$")
    ax.set_ylabel("$K$")
    return fig, ax


@optional_fig_ax
def show_activity_vs_opinion(
    opinions: np.ndarray,
    activities: np.ndarray,
    bins: int = 20,
    norm: Normalize = LogNorm(),
    ax: Axes = None,
    fig: Figure = None,
    title: str = "Density of activity and opinions",
    **kwargs,
) -> (Figure, Axes):
    """
    Density scatter plot colored by 2d histogram

    https://stackoverflow.com/a/53865762
    :param opinions: Array of opinions.
    :param activities: Array of activities, with indices corresponding to opninions
    :param bins: Number of bins to group opinions for determining density.
    :param norm: The scale of the density plot (normal, log, etc.)
    :param ax: Axes to use for plot. Created if none passed.
    :param fig: Figure to use for colorbar. Created if none passed.
    :param cbar_ax: Figure to use for colorbar. Created if none passed.
    :param title: Include a title for the axis.

    :keyword cbar_ax: Axes to use for plotting the colorbar (False to avoid flotting).

    :return: (Figure, Axes) used.
    """

    cbar_kws = {"label": P_A_X, **kwargs.pop("cbar_kws", {})}

    # get density based on bin size
    data, x_e, y_e = np.histogram2d(opinions, activities, bins=bins, density=True)
    # interpolate so shape is same as x and y
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([opinions, activities]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last

    idx = z.argsort()
    x, y, z = opinions[idx], activities[idx], z[idx]
    marker_size = kwargs.pop("s", 1)
    points = ax.scatter(x, y, s=marker_size, c=z, norm=norm, **kwargs)

    cax = cbar_kws.pop("cbar_ax", None) or cbar_kws.pop("cax", None)
    if cax is None:
        # create colorbar axes without stealing from main ax
        cbar = colorbar_inset(points, "outer right", size="5%", ax=ax, **cbar_kws,)
    elif isinstance(cax, Axes):
        # using existing cax
        cbar = fig.colorbar(points, cax=cax, **cbar_kws,)
    elif cax:
        # steal space from ax if cax is anything else (i.e. unless None, False, or an Axes)
        cbar = fig.colorbar(points, ax=ax, **cbar_kws)

    ax.set_xlabel(OPINION_SYMBOL)
    ax.set_ylabel(ACTIVITY_SYMBOL)
    ax.set_xlim(*get_equal_limits(opinions))
    if title:
        ax.set_title(title)
    return fig, ax, cbar


@optional_fig_ax
def show_matrix(
    mat: np.ndarray,
    label: str,
    map: str = "clustermap",
    sort: bool = False,
    norm: Normalize = LogNorm(),
    cmap: str = INTERACTIONS_CMAP,
    fig: Figure = None,
    ax: Axes = None,
    title: str = "matrix",
    **kwargs,
) -> (Figure or ClusterGrid, Axes):
    """
    Plot matrix where x and y are the indices of the matrix and z (or c) is the value.

    :param mat: Matrix to plot.
    :param label: The data represented by the matrix. E.g. Number of interactions.
    :param map: How to plot the matrix.
        * 'clustermap' - `sns.clustermap`
        * 'heatmap' - `sns.heatmap`
        * 'mesh' - `matplotlib.pcolormesh`
        * callable - calls the function using the (potentially generated) ax, norm, and keywords.
    :param sort: Sort the matrix by most interactions.
    :param norm: The scale of the plot (normal, log, etc.).
    :param cmap: Colormap to use.
    :param ax: Axes to use for plot. Created if none passed.
    :param fig: Figure to use for colorbar. Created if none passed.
    :param title: Include title in the figure.

    :keyword cbar_ax: Axes to use for plotting the colorbar.

    :return: (Figure, Axes) used.
    """

    N, M = mat.shape
    agent_mat = pd.DataFrame(
        mat,
        columns=pd.Index(np.arange(M), name="i"),
        index=pd.Index(np.arange(N), name="j"),
    )

    if sort:
        agent_mat = agent_mat.sort_values(
            by=list(agent_mat.index), axis="index"
        ).sort_values(by=list(agent_mat.columns), axis="columns")

    # default label for colorbar
    cbar_kws = {"label": label, **kwargs.pop("cbar_kws", {})}

    if "vmin" in kwargs:
        norm.vmin = kwargs["vmin"]
    if "vmax" in kwargs:
        norm.vmax = kwargs["vmax"]

    if map == "clustermap":
        if fig:
            plt.close(fig)
        fig = sns.clustermap(
            agent_mat, norm=norm, cmap=cmap, cbar_kws=cbar_kws, **kwargs
        )
        ax = fig.ax_heatmap
    elif map == "heatmap":
        sns.heatmap(
            agent_mat, norm=norm, cmap=cmap, ax=ax, cbar_kws=cbar_kws, **kwargs,
        )
        ax.invert_yaxis()
    elif map == "mesh":
        if sort:
            logger.warning(
                "'mesh' loses agent index information when sorting adjacency matrix"
            )
        mesh = ax.pcolormesh(agent_mat, norm=norm, cmap=cmap, **kwargs)
        ax.set_xlim(0, N)
        ax.set_ylim(0, M)
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
        map(agent_mat, ax=ax, norm=norm, **kwargs)
    else:
        raise NotImplementedError(
            f"Method {map} not implemented. Try one of 'clustermap' 'heatmap' 'mesh' or a "
            f"function."
        )
    ax.set_xlabel("Agent $i$")
    ax.set_ylabel("Agent $j$")
    if title:
        if map == "clustermap":
            fig.fig.suptitle(title)
        else:
            ax.set_title(title)
    return fig, ax


def show_noise_panel(
    df: pd.DataFrame, col, log=False, grid_kwargs=None, kde_kwargs=None
) -> sns.FacetGrid:
    if grid_kwargs is None:
        grid_kwargs = {}
    grid_kwargs.setdefault("palette", "husl")
    grid_kwargs.setdefault("hue", col)
    if kde_kwargs is None:
        kde_kwargs = {}
    kde_kwargs.setdefault("shade", True)
    kde_kwargs.setdefault("shade_lowest", False)

    _D = "D"
    if log:
        _D = "log D"
        df = df[df["D"] > 0]
        df[_D] = np.log10(df["D"])
    g = sns.FacetGrid(df, col=col, **grid_kwargs)
    g.map(sns.kdeplot, "opinion", _D, **kde_kwargs)
    g.axes[0, 0].set_ylabel(_D, rotation=0)
    return g
