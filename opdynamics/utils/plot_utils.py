import logging
from functools import partial
from operator import attrgetter
from typing import Iterable, Tuple

import numpy as np
from collections import Callable
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

logger = logging.getLogger("plot utils")


def colorbar_inset(
    mappable: ScalarMappable,
    position="outer right",
    size="2%",
    pad=0.0,
    orientation: str = None,
    ax: Axes = None,
    inset_axes_kwargs=None,
    **kwargs,
) -> Colorbar:
    """Create colorbar using axes toolkit by insetting the axis

    :param mappable:
    :param position:
    :param size:
    :param pad:
    :param orientation:
    :param ax:
    :param inset_axes_kwargs:

    :return: Color bar
    """
    ax = ax or mappable.axes
    fig = ax.figure
    if inset_axes_kwargs is None:
        if "outer" in position:
            inset_axes_kwargs = {}
        else:
            inset_axes_kwargs = {"borderpad": pad}
    if "top" in position or "bottom" in position and orientation is None:
        orientation = "horizontal"
    else:
        orientation = "vertical"

    if orientation == "vertical":
        height = "100%"
        width = size
    else:
        height = size
        width = "100%"
    if "outer" in position:
        # we use bbox to shift the colorbar across the entire image
        bbox = [0.0, 0.0, 1.0, 1.0]
        if "right" in position:
            loc = "center left"
            bbox[0] = 1.0 + pad
        elif "left" in position:
            loc = "center right"
            bbox[0] = -1.0 - pad
        elif "top" in position:
            loc = "lower left"
            bbox[1] = 1.0 + pad
        elif "bottom" in position:
            loc = "upper left"
            bbox[1] = -1.0 - pad
        else:
            raise ValueError(
                "unrecognised argument for 'position'. "
                "Valid locations are 'right' (default),'left','top', 'bottom' "
                "with each supporting 'inner' (default) and 'outer'"
            )
        ax_cbar = inset_axes(
            ax,
            width=width,
            height=height,
            loc=loc,
            bbox_to_anchor=bbox,
            bbox_transform=ax.transAxes,
            **inset_axes_kwargs,
        )
    else:
        ax_cbar = inset_axes(
            ax,
            width=width,
            height=height,
            loc=position.replace("inner", "").strip(),
            **inset_axes_kwargs,
        )
    return fig.colorbar(mappable, cax=ax_cbar, orientation=orientation, **kwargs)


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


def _make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def use_self_args(func: Callable = None, attrs: list = None, kwattrs: dict = None):
    """Replace function attributes (args and kwargs) with properties of a containing class.

    Example:
        def add(a, b):
            return a+b

        class AddTwo(object):
            def __init__(self):
                self.a = 2
            add = use_self_args(add, ["a"])


    """
    if not func:
        return partial(use_self_args, attrs=attrs, kwattrs=kwattrs)

    if attrs is None:
        attrs = []
    if kwattrs is None:
        kwattrs = {}

    def class_func(self, *args, **kwargs):
        """New function bound to a class."""

        # get attrs from self
        ec_args = []
        for ec_attr in attrs:
            ec_args.append(attrgetter(ec_attr)(self))
        # append normal provided args
        ec_args.extend(args)

        # get attrs from self and add to kwargs
        ec_kwargs = dict(**kwargs)
        for func_attr, ec_attr in kwattrs.items():
            ec_kwargs[func_attr] = attrgetter(ec_attr)(self)
        return func(*ec_args, **ec_kwargs)

    return class_func


def register_accessor(cls: object, func: Callable):
    """Add accessor to a class indirectly."""
    if hasattr(cls, func.__name__):
        logging.warning(
            f"{repr(func.__name__)} for type {repr(cls)} is overriding a preexisting"
            f"attribute with the same name.",
            UserWarning,
            stacklevel=2,
        )
    setattr(cls, func.__name__)
    return func


def get_equal_limits(values: Iterable) -> Tuple[float, float]:
    """Return 0-centered limits based on the absolute maximum value in values"""
    v = np.max(np.abs(values))
    return -v, v
