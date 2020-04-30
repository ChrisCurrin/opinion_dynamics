""""""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from opdynamics.echochamber import EchoChamber


class VisEchoChamber(object):
    def __init__(self, echochamber: EchoChamber):
        self.ec = echochamber

    def show_activities(self):
        ax = sns.distplot(self.ec.activities, kde=False, axlabel="activity")
        ax.set(title="Activity Distribution", ylabel="count")

    def show_opinions(self, color_code=True, ax=None):
        if ax is None:
            ax: Axes
            fig, ax = plt.subplots()

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
        ax.set_xlabel("$t$")
        ax.set_ylabel("$x_i{t}$")
        return ax


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


def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments
