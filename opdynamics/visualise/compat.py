"""Compatibility library"""
import numpy as np
import pandas as pd
import warnings

try:
    import statsmodels.nonparametric.api as smnp

    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False
from seaborn import (
    color_palette,
    light_palette,
    dark_palette,
    blend_palette,
    kdeplot as sns_kdeplot,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats


def _kde_support(data, bw, gridsize, cut, clip):
    """Establish support for a kernel density estimate."""
    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])
    support = np.linspace(support_min, support_max, gridsize)

    return support


def _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using scipy."""
    data = np.c_[x, y]
    kde = stats.gaussian_kde(data.T, bw_method=bw)
    data_std = data.std(axis=0, ddof=1)
    if isinstance(bw, str):
        bw = "scotts" if bw == "scott" else bw
        bw_x = getattr(kde, "%s_factor" % bw)() * data_std[0]
        bw_y = getattr(kde, "%s_factor" % bw)() * data_std[1]
    elif np.isscalar(bw):
        bw_x, bw_y = bw, bw
    else:
        msg = (
            "Cannot specify a different bandwidth for each dimension "
            "with the scipy backend. You should install statsmodels."
        )
        raise ValueError(msg)
    x_support = _kde_support(data[:, 0], bw_x, gridsize, cut, clip[0])
    y_support = _kde_support(data[:, 1], bw_y, gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


def _statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using statsmodels."""
    # statsmodels 0.8 fails on int type data
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    if isinstance(bw, str):
        bw_func = getattr(smnp.bandwidths, "bw_" + bw)
        x_bw = bw_func(x)
        y_bw = bw_func(y)
        bw = [x_bw, y_bw]
    elif np.isscalar(bw):
        bw = [bw, bw]

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    kde = smnp.KDEMultivariate([x, y], "cc", bw)
    x_support = _kde_support(x, kde.bw[0], gridsize, cut, clip[0])
    y_support = _kde_support(y, kde.bw[1], gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


def _bivariate_kdeplot(
    x,
    y,
    filled,
    fill_lowest,
    kernel,
    bw,
    gridsize,
    cut,
    clip,
    axlabel,
    cbar,
    cbar_ax,
    cbar_kws,
    ax,
    **kwargs,
):
    """Plot a joint KDE estimate as a bivariate contour plot."""
    # Determine the clipping
    if clip is None:
        clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
    elif np.ndim(clip) == 1:
        clip = [clip, clip]

    # Calculate the KDE
    if _has_statsmodels:
        xx, yy, z = _statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip)
    else:
        xx, yy, z = _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip)

    # Plot the contours
    n_levels = kwargs.pop("n_levels", 10)

    (scout,) = ax.plot([], [])
    default_color = scout.get_color()
    scout.remove()

    cmap = kwargs.pop("cmap", None)
    color = kwargs.pop("color", None)
    if cmap is None and "colors" not in kwargs:
        if color is None:
            color = default_color
        if filled:
            cmap = light_palette(color, as_cmap=True)
        else:
            cmap = dark_palette(color, as_cmap=True)
    if isinstance(cmap, str):
        if cmap.endswith("_d"):
            pal = ["#333333"]
            pal.extend(color_palette(cmap.replace("_d", "_r"), 2))
            cmap = blend_palette(pal, as_cmap=True)
        else:
            cmap = mpl.cm.get_cmap(cmap)

    label = kwargs.pop("label", None)

    kwargs["cmap"] = cmap
    contour_func = ax.contourf if filled else ax.contour
    cset = contour_func(xx, yy, z, n_levels, **kwargs)
    if filled and not fill_lowest:
        cset.collections[0].set_alpha(0)
    kwargs["n_levels"] = n_levels

    if cbar:
        cbar_kws = {} if cbar_kws is None else cbar_kws
        ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)

    # Label the axes
    if hasattr(x, "name") and axlabel:
        ax.set_xlabel(x.name)
    if hasattr(y, "name") and axlabel:
        ax.set_ylabel(y.name)

    if label is not None:
        legend_color = cmap(0.95) if color is None else color
        if filled:
            ax.fill_between([], [], color=legend_color, label=label)
        else:
            ax.plot([], [], color=legend_color, label=label)

    return ax


def kdeplot(
    *,
    x=None,
    y=None,
    shade=False,
    vertical=False,
    kernel="gau",
    bw="scott",
    gridsize=100,
    cut=3,
    clip=None,
    legend=True,
    cumulative=False,
    shade_lowest=True,
    cbar=False,
    cbar_ax=None,
    cbar_kws=None,
    ax=None,
    data=None,
    data2=None,  # TODO move data once * is enforced
    **kwargs,
):
    """Fit and plot a univariate or bivariate kernel density estimate.
    Parameters
    ----------
    x : 1d array-like
        Input data.
    y: 1d array-like, optional
        Second input data. If present, a bivariate KDE will be estimated.
    shade : bool, optional
        If True, shade in the area under the KDE curve (or draw with filled
        contours when data is bivariate).
    vertical : bool, optional
        If True, density is on x-axis.
    kernel : {'gau' | 'cos' | 'biw' | 'epa' | 'tri' | 'triw' }, optional
        Code for shape of kernel to fit with. Bivariate KDE can only use
        gaussian kernel.
    bw : {'scott' | 'silverman' | scalar | pair of scalars }, optional
        Name of reference method to determine kernel size, scalar factor,
        or scalar for each dimension of the bivariate plot. Note that the
        underlying computational libraries have different interperetations
        for this parameter: ``statsmodels`` uses it directly, but ``scipy``
        treats it as a scaling factor for the standard deviation of the
        data.
    gridsize : int, optional
        Number of discrete points in the evaluation grid.
    cut : scalar, optional
        Draw the estimate to cut * bw from the extreme data points.
    clip : pair of scalars, or pair of pair of scalars, optional
        Lower and upper bounds for datapoints used to fit KDE. Can provide
        a pair of (low, high) bounds for bivariate plots.
    legend : bool, optional
        If True, add a legend or label the axes when possible.
    cumulative : bool, optional
        If True, draw the cumulative distribution estimated by the kde.
    shade_lowest : bool, optional
        If True, shade the lowest contour of a bivariate KDE plot. Not
        relevant when drawing a univariate plot or when ``shade=False``.
        Setting this to ``False`` can be useful when you want multiple
        densities on the same Axes.
    cbar : bool, optional
        If True and drawing a bivariate KDE plot, add a colorbar.
    cbar_ax : matplotlib axes, optional
        Existing axes to draw the colorbar onto, otherwise space is taken
        from the main axes.
    cbar_kws : dict, optional
        Keyword arguments for ``fig.colorbar()``.
    ax : matplotlib axes, optional
        Axes to plot on, otherwise uses current axes.
    kwargs : key, value pairings
        Other keyword arguments are passed to ``plt.plot()`` or
        ``plt.contour{f}`` depending on whether a univariate or bivariate
        plot is being drawn.
    Returns
    -------
    ax : matplotlib Axes
        Axes with plot.
    See Also
    --------
    distplot: Flexibly plot a univariate distribution of observations.
    jointplot: Plot a joint dataset with bivariate and marginal distributions.
    Examples
    --------
    Plot a basic univariate density:
    .. plot::
        :context: close-figs
        >>> import numpy as np; np.random.seed(10)
        >>> import seaborn as sns; sns.set(color_codes=True)
        >>> mean, cov = [0, 2], [(1, .5), (.5, 1)]
        >>> x, y = np.random.multivariate_normal(mean, cov, size=50).T
        >>> ax = sns.kdeplot(x=x)
    Shade under the density curve and use a different color:
    .. plot::
        :context: close-figs
        >>> ax = sns.kdeplot(x=x, shade=True, color="r")
    Plot a bivariate density:
    .. plot::
        :context: close-figs
        >>> ax = sns.kdeplot(x=x, y=y)
    Use filled contours:
    .. plot::
        :context: close-figs
        >>> ax = sns.kdeplot(x=x, y=y, shade=True)
    Use more contour levels and a different color palette:
    .. plot::
        :context: close-figs
        >>> ax = sns.kdeplot(x=x, y=y, n_levels=30, cmap="Purples_d")
    Use a narrower bandwith:
    .. plot::
        :context: close-figs
        >>> ax = sns.kdeplot(x=x, bw=.15)
    Plot the density on the vertical axis:
    .. plot::
        :context: close-figs
        >>> ax = sns.kdeplot(x=y, vertical=True)
    Limit the density curve within the range of the data:
    .. plot::
        :context: close-figs
        >>> ax = sns.kdeplot(x=x, cut=0)
    Add a colorbar for the contours:
    .. plot::
        :context: close-figs
        >>> ax = sns.kdeplot(x=x, y=y, cbar=True)
    Plot two shaded bivariate densities:
    .. plot::
        :context: close-figs
        >>> iris = sns.load_dataset("iris")
        >>> setosa = iris.loc[iris.species == "setosa"]
        >>> virginica = iris.loc[iris.species == "virginica"]
        >>> ax = sns.kdeplot(x=setosa.sepal_width, y=setosa.sepal_length,
        ...                  cmap="Reds", shade=True, shade_lowest=False)
        >>> ax = sns.kdeplot(x=virginica.sepal_width, y=virginica.sepal_length,
        ...                  cmap="Blues", shade=True, shade_lowest=False)
    """
    # Handle deprecation of `data` as name for x variable
    # TODO this can be removed once refactored to do centralized preprocessing
    # of input variables, because a vector input to `data` will be treated like
    # an input to `x`. Warning is probably not necessary.
    x_passed_as_data = x is None and data is not None and np.ndim(data) == 1
    if x_passed_as_data:
        x = data

    # Handle deprecation of `data2` as name for y variable
    if data2 is not None:
        msg = "The `data2` param is now named `y`; please update your code."
        warnings.warn(msg)
        y = data2

    # TODO replace this preprocessing with central refactoring
    if isinstance(x, list):
        x = np.asarray(x)
    if isinstance(y, list):
        y = np.asarray(y)

    bivariate = x is not None and y is not None
    if bivariate and cumulative:
        raise TypeError(
            "Cumulative distribution plots are not"
            "supported for bivariate distributions."
        )

    if ax is None:
        ax = plt.gca()

    if bivariate:
        ax = _bivariate_kdeplot(
            x,
            y,
            shade,
            shade_lowest,
            kernel,
            bw,
            gridsize,
            cut,
            clip,
            legend,
            cbar,
            cbar_ax,
            cbar_kws,
            ax,
            **kwargs,
        )
    else:
        sns_kdeplot(
            x=x,
            y=y,
            vertical=vertical,
            fill=shade,
            thresh=int(shade_lowest),
            bw_adjust=bw,
            gridsize=gridsize,
            cut=cut,
            clip=clip,
            legend=legend,
            cumulative=cumulative,
            cbar=cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            ax=ax,
            data=data,
            data2=data2,
            **kwargs,
        )

    return ax


# re-assign
import seaborn

seaborn.kdeplot = kdeplot
