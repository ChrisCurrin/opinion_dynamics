import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from scipy.interpolate import interpn

from opdynamics.utils.constants import ABS_MEAN_FINAL_OPINION
from opdynamics.utils.constants import ACTIVITY_SYMBOL, OPINION_SYMBOL, P_A_X
from opdynamics.utils.decorators import optional_fig_ax
from opdynamics.utils.plot_utils import get_equal_limits, colorbar_inset


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
    title: bool = True,
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
        colorbar_inset(
            points, "outer right", size="5%", ax=ax, **cbar_kws,
        )
    elif isinstance(cax, Axes):
        # using existing cax
        fig.colorbar(
            points, cax=cax, **cbar_kws,
        )
    elif cax:
        # steal space from ax if cax is anything else (i.e. unless None, False, or an Axes)
        fig.colorbar(points, ax=ax, **cbar_kws)

    ax.set_xlabel(OPINION_SYMBOL)
    ax.set_ylabel(ACTIVITY_SYMBOL)
    ax.set_xlim(*get_equal_limits(opinions))
    if title:
        ax.set_title("Density of activity and opinions")
    return fig, ax
