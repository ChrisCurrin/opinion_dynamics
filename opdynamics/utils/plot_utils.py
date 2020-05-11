from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def colorbar_inset(
    mappable: ScalarMappable,
    position="outer right",
    size="2%",
    orientation: str = None,
    ax: Axes = None,
    inset_axes_kwargs=None,
    **kwargs
) -> Colorbar:
    """Create colorbar using axes toolkit by insetting the axis

    :param mappable:
    :param position:
    :param size:
    :param orientation:
    :param ax:
    :param inset_axes_kwargs:

    :return: Color bar
    """
    ax = ax or mappable.axes
    fig = ax.figure
    if inset_axes_kwargs is None:
        inset_axes_kwargs = {"borderpad": 0.0}
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
            bbox[0] = 1.0
        elif "left" in position:
            loc = "center right"
            bbox[0] = -1.0
        elif "top" in position:
            loc = "lower left"
            bbox[1] = 1.0
        elif "bottom" in position:
            loc = "upper left"
            bbox[1] = -1.0
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
            **inset_axes_kwargs
        )
    else:
        ax_cbar = inset_axes(
            ax,
            width=width,
            height=height,
            loc=position.replace("inner", "").strip(),
            **inset_axes_kwargs
        )
    return fig.colorbar(mappable, cax=ax_cbar, orientation=orientation, **kwargs)
