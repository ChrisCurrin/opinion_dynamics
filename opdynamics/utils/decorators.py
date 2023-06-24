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


class timeblock:
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
        with timeblock:
            return f(*args, **kwargs)

    return wrapper


def hashable(cls):
    """Make a class hashable by using its `__str__` (or `__repr__` if there's no `__str__`) method.
    This generates the same hash for the same input across sessions.
    (Python, by default, does not due to security reasons).

    Properties
    ======
    - __hash__
    = hash_extra

    Examples
    ======

    .. code-block :: python

        @hashable
        class MyClass(object):
            def __init__(self, a):
                self.a = a
            def __str__(self):
                return f"MyClass({a})"
        hash(MyClass("first a"))==hash(MyClass("first a")) # True
        hash(MyClass("first a"))==hash(MyClass("second a")) # False

    """
    import hashlib

    def __hash__(self):
        return int(hashlib.sha256(str(self).encode("utf-8")).hexdigest(), 16)

    def hash_extra(self, extra=""):
        full_str = str(self) + extra
        return int(hashlib.sha256(full_str.encode("utf-8")).hexdigest(), 16)

    cls.__hash__ = __hash__
    cls.hash_extra = hash_extra
    return cls


def hash_repeat(*args, **kwargs):
    """Given the same arguments, the same number is produced."""
    import hashlib
    import json

    return int(
        hashlib.sha256(
            (str(args) + str(json.dumps(kwargs, sort_keys=True))).encode("utf-8")
        ).hexdigest(),
        16,
    )


class CountCalls:
    """Simple counter, counting an invocation my means of a string (explicit call) or function name (decorator)

    see: https://gist.github.com/AndiH/ba697e888fb4d92f37b31983949fe17c

    .. code-block :: python

        @Counter
        def testfunction():
            return 1

        def example():
            Counter("Start")
            Counter("Start")
            Counter("End")

            testfunction()
            testfunction()
            testfunction()

            Counter.print()

    """

    counters = {}

    def __init__(self, *args, **kwargs):
        """Construct class. First, check if class is called via a decorator or directly. If true, defer incrementing counter until wrapped function is actually called (see __call__). If false, increment counter now."""
        if callable(args[0]):
            self.func = args[0]
            self.name = args[0].__name__
        else:
            name = args[0]

            self.potentiallyIncrementByName(name)

    def __call__(self, *args, **kwargs):
        """Function for when Counter is used as a decorator. In that case, this class method call with increment the counter."""
        self.potentiallyIncrementByName(self.name)
        return self.func(*args, **kwargs)

    @classmethod
    def potentiallyIncrementByName(self, name):
        """Either the Counter is incremented (if it exists already) or created"""
        CountCalls.counters[name] = CountCalls.counters.get(name, 0) + 1

    @classmethod
    def print(self):
        """Convenience method to print counted methods."""
        padding = len(max(CountCalls.counters.keys()))
        for name, count in CountCalls.counters.items():
            print(f"{name:>{padding}}: {count}")
