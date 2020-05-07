from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def optional_fig_ax(function):
    """
    Decorator that optionally creates a new figure and axis if they are not defined during method call.
    """

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
        function(*args, **kwargs)

    return wrapper
