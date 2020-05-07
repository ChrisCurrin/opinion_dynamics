""""""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from matplotlib.colors import LogNorm, Normalize
from scipy.interpolate import interpn

from opdynamics.echochamber import EchoChamber
from opdynamics.utils.constants import (
    ACTIVITY_SYMBOL,
    OPINION_AGENT_TIME,
    OPINION_SYMBOL,
    P_A_X,
    TIME_SYMBOL,
)
from opdynamics.utils.decorators import optional_fig_ax


class VisEchoChamber(object):
    def __init__(self, echochamber: EchoChamber):
        self.ec = echochamber

    @optional_fig_ax
    def show_activities(self, fig=None, ax=None):
        sns.distplot(self.ec.activities, kde=False, axlabel="activity", ax=ax)
        ax.set(title="Activity Distribution", ylabel="count")
        return fig, ax

    @optional_fig_ax
    def show_opinions(self, color_code=True, ax=None, fig=None):
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
        ax.set_xlabel(TIME_SYMBOL)
        ax.set_ylabel(OPINION_AGENT_TIME)
        return fig, ax

    @optional_fig_ax
    def show_activity_vs_opinions(
        self, bins=20, norm=LogNorm(), ax=None, fig=None, **kwargs
    ):
        """
        Density scatter plot colored by 2d histogram

        https://stackoverflow.com/a/53865762
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

        points = ax.scatter(x, y, c=z, norm=norm, **kwargs)

        fig.colorbar(points, ax=ax, label=P_A_X)

        ax.set_xlabel(OPINION_SYMBOL)
        ax.set_ylabel(ACTIVITY_SYMBOL)

        return ax


# Methods which may be generally useful.


def colorline(
    x,
    y,
    z=None,
    cmap="seismic",
    norm=Normalize(0.0, 1.0),
    linewidth=1.0,
    ax=None,
    **kwargs
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
    fig.colorbar(im)
    ax.set_xlabel()
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
