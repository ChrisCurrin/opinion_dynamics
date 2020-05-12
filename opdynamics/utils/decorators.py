import time
import logging
from functools import wraps
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Callable


def optional_fig_ax(function: Callable):
    """Decorator that optionally creates a new figure and axis if they are not defined during method call.

    Usage:
    @optional_fig_ag
    def function(ax=None, fig=None):
        ...
        return fig, ax
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        """
        If neither `fig` nor `ax` are defined, then call plt.subplots().

        If both `fig` and `ax` are defined, pass them on.

        If only `ax` is defined, retrieve `fig` from the passed `ax`

        If only `fig` is defined, retrieve the current axis from `figure` using `gca()`
        """
        fig: Figure = kwargs.pop("fig", None)
        ax: Axes = kwargs.pop("ax", None)
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        elif fig is None:
            fig = ax.figure
        elif ax is None:
            ax = fig.gca()
        kwargs["fig"] = fig
        kwargs["ax"] = ax
        return function(*args, **kwargs)

    return wrapper


class timeblock(object):
    """Context guard for timing a block of code"""

    def __enter__(self, name=""):
        self.start = time.time()
        self.name = name

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        logging.debug(f"{self.name} took: {end - self.start:2.2f} sec")


def timefunc(f):
    """Decorator for timing functions
    Usage:
    @timing
    def function(a):
        pass
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logging.debug(f"function:{f.__name__} took: {end-start:2.2f} sec")
        return result

    return wrapper
