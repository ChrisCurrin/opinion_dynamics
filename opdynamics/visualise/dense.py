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
    # default to borders
    sns.despine(ax=ax, top=False, right=False, left=False, bottom=False)
    return fig, ax


def show_jointplot(
    x, y, ax=(), cmap=None, joint_kws=None, marginal_kws=None, annot_kws=None, **kwargs
):
    """Nearest Neighbour plot for inside a bigger figure"""

    ax_joint, ax_marg_x, ax_marg_y = ax
    # Set up empty default kwarg dicts
    joint_kws = {} if joint_kws is None else joint_kws.copy()
    joint_kws.update(kwargs)
    marginal_kws = {} if marginal_kws is None else marginal_kws.copy()
    annot_kws = {} if annot_kws is None else annot_kws.copy()

    # Make a colormap based off the plot color
    if cmap is None:
        cmap = sns.cubehelix_palette(8, reverse=True, as_cmap=True)
        color = sns.cubehelix_palette(8, reverse=True)[3]

    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Turn off the ticks on the density axis for the marginal plots
    plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)
    ax_marg_x.yaxis.grid(False)
    ax_marg_y.xaxis.grid(False)

    # Convert the x and y data to arrays for indexing and plotting
    x_array = np.asarray(x)
    y_array = np.asarray(y)

    # Possibly drop NA
    not_na = pd.notnull(x_array) & pd.notnull(y_array)
    x_array = x_array[not_na]
    y_array = y_array[not_na]

    joint_kws.setdefault("shade", True)
    joint_kws.setdefault("cmap", cmap)
    sns.kdeplot(x, y, ax=ax_joint, **joint_kws)

    marginal_kws.setdefault("shade", True)
    marginal_kws.setdefault("color", color)
    sns.kdeplot(x, vertical=False, ax=ax_marg_x, **marginal_kws)
    sns.kdeplot(y, vertical=True, ax=ax_marg_y, **marginal_kws)


def show_noise_panel(
    df: pd.DataFrame,
    col,
    log=False,
    grid_kwargs=None,
    kde_kwargs=None,
    palette_kwargs=None,
    fig: Figure = None,
    ax: "np.ndarray[Axes]" = None,
) -> (Figure, "np.ndarray[Axes]"):
    """Display a grid of kernel density estimates (nudge vs opinion) for different parameters."""

    if grid_kwargs is None:
        grid_kwargs = {}
    if kde_kwargs is None:
        kde_kwargs = {}
    kde_kwargs.setdefault("shade", True)
    kde_kwargs.setdefault("shade_lowest", False)
    if palette_kwargs is None:
        palette_kwargs = {}
    start = palette_kwargs.pop("start", 0)
    n_colors = kde_kwargs.setdefault("levels", 6)

    col_names = df[col].unique()
    not_na = ~df[col].isnull()

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, len(col_names), **grid_kwargs)
    elif fig is None:
        fig = ax[0].figure

    _D = "D"
    if log:
        _D = "log D"
        df = df[df["D"] > 0]
        df[_D] = np.log10(df["D"])

    col_masks = [df[col] == n for n in col_names]
    hues = [
        sns.cubehelix_palette(
            n_colors=n_colors,
            start=start + 3 * (n - start) / len(col_names),
            as_cmap=True,
            **palette_kwargs,
        )
        for n in range(len(col_names))
    ]
    for j, (col_mask, hue) in enumerate(zip(col_masks, hues)):
        data_ijk = df[not_na & col_mask]
        sns.kdeplot(data_ijk["opinion"], data_ijk[_D], ax=ax[j], cmap=hue, **kde_kwargs)
    for _ax in ax[1:]:
        _ax.set_ylabel("")
    for col_name, _ax in zip(col_names, ax):
        _ax.set_title(col_name)
    ax[0].set_ylabel(_D, rotation=0)
    return fig, ax


def show_opinion_grid(
    df: pd.DataFrame, columns: list, grid_kwargs=None, kde_kwargs=None
) -> sns.FacetGrid:
    """Plot a grid of noise vs opinion kernel density estimates where columns and rows of the grid are
    different combinations of parameters (according to `columns`).

    Seaborn's ``sns.FacetGrid`` is used.

    :param df: Long-form DataFrame of observations.
    :param columns: The column names in df to construct the FacetGrid. The first column changes with the co'l' of
        grid, the second changes with the 'row' of the grid, and the final column changes with the 'hue' of the grid.
    :param grid_kwargs: Keyword arguments for ``sns.FacetGrid``.
    :param kde_kwargs: Keyword arguments for ``sns.kdeplot``.
    :return: FaceGrid of kernel density estimates for diffrent parameter combinations.
    """

    if grid_kwargs is None:
        grid_kwargs = {}
    if kde_kwargs is None:
        kde_kwargs = {}

    data_kwargs = dict(zip(["col", "row", "hue"], columns))

    g = sns.FacetGrid(df, **data_kwargs, **grid_kwargs)

    if "hue" not in data_kwargs:
        kde_kwargs.setdefault("cmap", sns.cubehelix_palette(reverse=True, as_cmap=True))
    kde_kwargs.setdefault("shade", True)
    kde_kwargs.setdefault("shade_lowest", False)
    if "D" in columns:
        cmap = kde_kwargs.pop("cmap", sns.cubehelix_palette(reverse=True, as_cmap=True))
        g.map(sns.kdeplot, "opinion", **kde_kwargs)
    else:
        g.map(sns.kdeplot, "opinion", "D", **kde_kwargs)
    return g
