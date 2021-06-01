""""""
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Union

from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from opdynamics.socialnetworks import SocialNetwork
from opdynamics.utils.constants import *
from opdynamics.utils.decorators import optional_fig_ax
from opdynamics.utils.plot_utils import get_time_point_idx, use_self_args
from opdynamics.utils.plot_utils import colorbar_inset, colorline
from opdynamics.visualise.dense import show_activity_vs_opinion, show_matrix

logger = logging.getLogger("visualise")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


class VisSocialNetwork(object):
    """Class used to visualise attached SocialNetwork object.

    Visualisations
    =====
    - ``show_activities``
    - ``show_opinions``
    - ``show_opinions_snapshot``
    - ``show_agent_opinions``
    - ``show_adjacency_matrix``
    - ``show_nearest_neighbour``
    - ``show_summary``


    Private methods
    ----
     - ``_get_equal_opinion_limits``

    Examples
    =====

    Normal usage
    -----

    .. code-block :: python

        vis = VisSocialNetwork(sn)
        vis.show_opinions()

    One-time use
    -----

    .. code-block :: python

        VisSocialNetwork(sn).show_opinions()

    """

    def __init__(self, SocialNetwork: SocialNetwork):
        self.sn = SocialNetwork

    show_activity_vs_opinion = use_self_args(
        show_activity_vs_opinion, ["sn.opinions", "sn.activities"]
    )

    def show_connection_probabilities(
        self, *args, title="Connection probabilities", **kwargs
    ) -> Tuple[Figure, Axes]:
        """
        Plot connection probability matrix.

        :keyword map: How to plot the matrix.
                * 'clustermap' - `sns.clustermap`
                * 'heatmap' - `sns.heatmap`
                * 'mesh' - `matplotlib.pcolormesh`
                * callable - calls the function using the (potentially generated) ax, norm, and keywords.
        :keyword sort: Sort the matrix by most interactions.
        :keyword norm: The scale of the plot (normal, log, etc.).
        :keyword cmap: Colormap to use.
        :keyword ax: Axes to use for plot. Created if none passed.
        :keyword fig: Figure to use for colorbar. Created if none passed.
        :keyword title: Include title in the figure.

        :keyword cbar_ax: Axes to use for plotting the colorbar.

        :return: Tuple[Figure, Axes] used.

        """
        cmap = kwargs.pop("cmap", CONNECTIONS_CMAP)
        # noinspection PyProtectedMember
        p_conn = self.sn.adj_mat._p_conn
        if p_conn is None:
            p_conn = self.sn.adj_mat.conn_method(self.sn, **self.sn.adj_mat.conn_kwargs)
        return show_matrix(
            p_conn,
            "$P_{ij}$",
            *args,
            # min value must be > 0 when LogNorm is used
            vmin=np.min(p_conn[p_conn > 0]),
            vmax=1,
            cmap=cmap,
            title=title,
            **kwargs,
        )

    def show_adjacency_matrix(
        self,
        *args,
        title="Cumulative adjacency matrix",
        t: Union[Tuple[Union[int, float]], int, float] = -1,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot adjacency matrix.

        If matrix `sn.adj_mat` is 3D (first dimension is time), the sum of the interactions is computed over time.
        The total adjacency matrix

        > Note adj_mat is indexed ji but represents Aij (if there is input from agent j to agent i)

        :keyword map: How to plot the matrix.
                * 'clustermap' - `sns.clustermap`
                * 'heatmap' - `sns.heatmap`
                * 'mesh' - `matplotlib.pcolormesh`
                * callable - calls the function using the (potentially generated) ax, norm, and keywords.
        :keyword sort: Sort the matrix by most interactions.
        :keyword norm: The scale of the plot (normal, log, etc.).
        :keyword cmap: Colormap to use.
        :keyword ax: Axes to use for plot. Created if none passed.
        :keyword fig: Figure to use for colorbar. Created if none passed.
        :keyword title: Include title in the figure.

        :keyword cbar_ax: Axes to use for plotting the colorbar.

        :return: Tuple[Figure, Axes] used.

        """
        cmap = kwargs.pop("cmap", INTERACTIONS_CMAP)

        t_idx = get_time_point_idx(self.sn.result.t, t)

        conn_weights = self.sn.adj_mat.accumulate(t_idx)

        return show_matrix(
            conn_weights,
            "Number of interactions",
            *args,
            vmin=1,
            cmap=cmap,
            title=title,
            **kwargs,
        )

    # alias
    show_interactions = show_adjacency_matrix

    @optional_fig_ax
    def show_activities(
        self, ax: Axes = None, fig: Figure = None, **kwargs
    ) -> Tuple[Figure, Axes]:
        kwargs.setdefault("color", "Green")
        sns.histplot(self.sn.activities, ax=ax, **kwargs)
        ax.set(
            title="Activity distribution",
            ylabel="count",
            xlabel="activity",
            xlim=(
                np.min(np.append(self.sn.activities, 0)),
                np.max(np.append(self.sn.activities, 1)),
            ),
        )
        return fig, ax

    @optional_fig_ax
    def show_opinions(
        self,
        color_code=True,
        subsample: int = 1,
        ax: Axes = None,
        fig: Figure = None,
        title: str = "Opinion dynamics",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Display the evolution of opinions over time.

        :param color_code: Whether to color by valence (True) or by agent (False).
            By default, a scatter plot is used, but a line plot with precise coloring can be specified with 'line'.
        :param subsample: The number of agents to plot.
            A subsample of 1 means every agent is plotted. 10 is every 10th agent, etc.
        :param ax: Axes to use for plot. Created if none passed.
        :param fig: Figure to use for colorbar. Created if none passed.
        :param title: Include title in the figure.

        :keyword cmap: Colormap to use (color_code = True or 'line')
        :keyword vmin: minimum value to color from cmap. Lower values will be colored the same.
        :keyword vmax: maximum value to color from cmap. Higher values will be colored the same.

        :return: Tuple[Figure, Axes] used.
        """
        cmap = kwargs.pop("cmap", OPINIONS_CMAP)
        vmin = kwargs.pop("vmin", np.min(self.sn.result.y))
        vmax = kwargs.pop("vmax", np.max(self.sn.result.y))
        lw = kwargs.pop("lw", 0.1)
        sm = ScalarMappable(norm=TwoSlopeNorm(0, vmin, vmax), cmap=cmap)
        import pandas as pd

        df_opinions: pd.DataFrame = self.sn.result_df().iloc[::subsample]

        if color_code == "line" or color_code == "lines":
            # using the colorline method allows colors to be dependent on a value, in this case, opinion,
            # but takes much longer to display
            for agent_idx, agent_opinions in df_opinions.iteritems():
                c = sm.to_rgba(agent_opinions.values)
                colorline(
                    agent_opinions.index,
                    agent_opinions.values,
                    c,
                    lw=lw,
                    ax=ax,
                    **kwargs,
                )
        elif color_code:
            for agent_idx, agent_opinions in df_opinions.iteritems():
                c = sm.to_rgba(agent_opinions.values)
                s = kwargs.pop("s", 0.1)
                ax.scatter(
                    agent_opinions.index, agent_opinions.values, c=c, s=s, **kwargs
                )
        else:
            ls = kwargs.pop("ls", "-")
            mec = kwargs.pop("mec", "None")
            with sns.color_palette("Set1", n_colors=df_opinions.shape[0]):
                ax.plot(
                    df_opinions,
                    ls=ls,
                    mec=mec,
                    lw=lw,
                    **kwargs,
                )
        ax.set_xlim(0, df_opinions.index[-1])
        ax.set_xlabel(TIME_SYMBOL)
        ax.set_ylabel(OPINION_AGENT_TIME)
        ax.set_ylim(*self._get_equal_opinion_limits())
        if title:
            ax.set_title(title)
        return fig, ax

    @optional_fig_ax
    def show_opinions_change(
        self,
        t: float = 0,
        ax: Axes = None,
        fig: Figure = None,
        title: str = "Opinion dynamics (with change by color)",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Display the evolution of opinions over time.

        :param t: The time point to consider the change of opinion from/to. Integers are treated as indices.
            Default is 0 (the first time point).
        :param ax: Axes to use for plot. Created if none passed.
        :param fig: Figure to use for colorbar. Created if none passed.
        :param title: Include title in the figure.

        :keyword cmap: Colormap to use (color_code = True or 'line')
        :keyword vmin: minimum value to color from cmap. Lower values will be colored the same.
        :keyword vmax: maximum value to color from cmap. Higher values will be colored the same.

        :return: Tuple[Figure, Axes] used.
        """

        cmap = kwargs.pop("cmap", "viridis")
        df_opinions = self.sn.result_df()

        idx = get_time_point_idx(self.sn.result.t, t)

        t_val = df_opinions.index[idx]
        df_change_from_start = np.abs(df_opinions.iloc[idx] - df_opinions)

        vmin = kwargs.pop("vmin", np.max([df_change_from_start.values.min(), 1]))
        vmax = kwargs.pop("vmax", df_change_from_start.values.max())

        sm = ScalarMappable(norm=LogNorm(vmin, vmax), cmap=cmap)

        # using the colorline method allows colors to be dependent on a value, in this case, opinion,
        # but takes much longer to display
        for (_, agent_opinion), (_, dagent_opinion) in zip(
            df_opinions.iteritems(), df_change_from_start.iteritems()
        ):
            # add vmin for 0 values (for valid log(x) in LogNorm)
            mask_zero = dagent_opinion.values == 0
            c = sm.to_rgba(dagent_opinion.values + mask_zero * sm.norm.vmin)
            lw = 0.1
            colorline(
                agent_opinion.index,
                agent_opinion.values,
                c,
                lw=lw,
                ax=ax,
            )
            # ax.scatter(
            #     agent_opinion.index, agent_opinion.values, c=c, s=s,
            # )

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(math_fix(f"$d{OPINION_SYMBOL}_{{i, t={t_val:.2f}}}(t)$"))
        ax.set_xlim(0, df_opinions.index[-1])
        ax.set_xlabel(TIME_SYMBOL)
        ax.set_ylabel(OPINION_AGENT_TIME)
        ax.set_ylim(*self._get_equal_opinion_limits())
        if title:
            ax.set_title(title)
        return fig, ax

    @optional_fig_ax
    def show_opinions_distribution(
        self,
        t=-1,
        ax: Axes = None,
        fig: Figure = None,
        title: str = "Opinions distribution",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        idx = get_time_point_idx(self.sn.result.t, t)
        bins = kwargs.pop("bins", "auto")
        kwargs.setdefault("color", "Purple")
        kwargs.setdefault("kde", True)
        kwargs.setdefault("stat", "probability")
        vertical = kwargs.pop("vertical", False)

        data = {
            "x": self.sn.result.y[:, idx] if not vertical else None,
            "y": self.sn.result.y[:, idx] if vertical else None,
        }
        sns.histplot(**data, bins=bins, ax=ax, **kwargs)

        if vertical:
            ax.set_ylabel(OPINION_SYMBOL)
            ax.set_xlabel(f"$P({OPINION_SYMBOL})$")
            ax.set_ylim(*self._get_equal_opinion_limits())
        else:
            ax.set_xlabel(OPINION_SYMBOL)
            ax.set_ylabel(f"$P({OPINION_SYMBOL})$")
            ax.set_xlim(*self._get_equal_opinion_limits())
        if title:
            ax.set_title(title)
        return fig, ax

    @optional_fig_ax
    def show_agent_opinions(
        self,
        t=-1,
        direction=True,
        sort=False,
        ax: Axes = None,
        fig: Figure = None,
        colorbar: bool = True,
        title: str = "Agent opinions",
        show_middle=True,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        cmap = kwargs.pop("cmap", OPINIONS_CMAP)

        idx = get_time_point_idx(self.sn.result.t, t)
        opinions = self.sn.result.y[:, idx]
        agents = np.arange(self.sn.N)
        if not direction:
            # only magnitude
            opinions = np.abs(opinions)

        if np.iterable(sort) or sort:

            if isinstance(sort, np.ndarray):
                # sort passed as indices
                ind = sort
            else:
                logger.warning(
                    "sorting opinions for `show_agent_opinions` means agent indices are jumbled"
                )
                # sort by opinion
                ind = np.argsort(opinions)
            opinions = opinions[ind]

        v = self._get_equal_opinion_limits()
        sm = ScalarMappable(norm=Normalize(*v), cmap=cmap)
        color = sm.to_rgba(opinions)

        ax.barh(
            agents,
            opinions,
            color=color,
            edgecolor="None",
            linewidth=0,  # remove bar borders
            height=1,  # per agent
            **kwargs,
        )
        ax.axvline(x=0, ls="-", color="k", alpha=0.5, lw=1)
        if (np.iterable(sort) or sort) and show_middle:
            min_idx = np.argmin(np.abs(opinions))
            ax.hlines(
                y=min_idx,
                xmin=v[0],
                xmax=v[1],
                ls="--",
                color="k",
                alpha=0.5,
                lw=1,
            )
            ax.annotate(
                f"{min_idx}",
                xy=(np.min(opinions), min_idx),
                fontsize="small",
                color="k",
                alpha=0.5,
                va="bottom",
                ha="left",
            )
        if colorbar:
            # create colorbar axes without stealing from main ax
            cbar = colorbar_inset(sm, "outer bottom", size="5%", pad=0.01, ax=ax)
            sns.despine(ax=ax, bottom=True)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            cbar.set_label(OPINION_SYMBOL)

        ax.set_ylim(0, self.sn.N)
        ax.set_xlim(*v)
        if not colorbar:
            # xlabel not part of colorbar
            ax.set_xlabel(OPINION_SYMBOL)
        ax.set_ylabel("Agent $i$")
        ax.yaxis.set_major_locator(MaxNLocator(5))
        if title:
            ax.set_title(title)
        return fig, ax

    def _get_equal_opinion_limits(self):
        if self.sn.result is None:
            opinions = self.sn._opinions
        else:
            opinions = self.sn.result.y
        v = np.max(np.abs(opinions))
        return -v, v

    def show_nearest_neighbour(
        self, bw_adjust=0.5, t=-1, title=True, **kwargs
    ) -> sns.JointGrid:
        nn = self.sn.get_nearest_neighbours(t)
        idx = get_time_point_idx(self.sn.result.t, t)
        opinions = self.sn.result.y[:, idx]
        kwargs.setdefault("color", "Purple")
        marginal_kws = kwargs.pop("marginal_kws", dict())
        marginal_kws.update(bw_adjust=bw_adjust)
        g = sns.jointplot(
            self.sn._opinions,
            nn,
            kind="kde",
            bw_adjust=bw_adjust,
            marginal_kws=marginal_kws,
            **kwargs,
        )
        g.ax_joint.set_xlabel(OPINION_SYMBOL)
        g.ax_joint.set_ylabel(MEAN_NEAREST_NEIGHBOUR)
        if title:
            g.fig.suptitle("Neighbour's opinions", va="bottom")
        return g

    def show_graph(self, **kwargs):
        import networkx as nx

        cmap = kwargs.pop("cmap", "bwr")
        vmin = kwargs.pop("vmin", np.min(self.sn._opinions))
        vmax = kwargs.pop("vmax", np.max(self.sn._opinions))
        alpha = kwargs.pop("alpha", 0.5)

        G = self.sn.get_network_graph()

        edge_weights = [d["weight"] for (u, v, d) in G.edges(data=True)]
        scale_weight = 1 / np.max(edge_weights)

        sm = ScalarMappable(norm=TwoSlopeNorm(0, vmin, vmax), cmap=cmap)
        edge_sm = ScalarMappable(
            norm=LogNorm(max(min(edge_weights), 1), np.max(edge_weights)),
            cmap="viridis",
        )
        c = sm.to_rgba(self.sn._opinions)
        edge_c = edge_sm.to_rgba(self.sn)
        # ajdust the alpha
        c[:, 3] = alpha
        edge_c[:, 3] = alpha

        # layout
        logger.debug("laying out using fruchterman_reingold_layout...")
        # use fruchterman_reingold_layout
        # see https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.spring_layout.html

        # # Initial position of agents
        # initial_pos = (
        #     df_degree.sort_values(by="opinion")["opinion"]
        #     .apply(
        #         lambda x: (
        #             (-5 if x < 0 else 5) + np.random.uniform(-1, 1),
        #             np.random.uniform(-1, 1),
        #         )
        #     )
        #     .to_dict()
        # )
        pos = nx.spring_layout(
            G,
            center=(0, 0),
            iterations=int(self.sn.N * np.log10(self.sn.N)),
            weight="weight",
            seed=1337,
        )

        logger.debug("drawing graph...")

        nx.draw_networkx(
            G,
            pos,
            with_labels=False,
            node_color=c,
            node_size=10,
            arrowsize=0.3,
            linewidths=np.array(edge_weights) * scale_weight,
            width=np.array(edge_weights) * scale_weight,
            edge_color=edge_c,
        )

        logger.debug("done drawing graph")

        # logger.debug("*"*32+"\nsaving graph as png...")
        # plt.savefig("output/fig_net.png", dpi=300)
        # logger.debug("*"*32+"\nsaving graph as pdf...")
        # plt.savefig("output/fig_net.pdf", dpi=300, rasterized=True)
        return plt.gcf()

    def show_summary(self, single_fig=True, fig_kwargs=None) -> Tuple[Figure, Axes]:
        nrows = 3
        ncols = 2
        if single_fig:
            if fig_kwargs is None:
                fig_kwargs = {
                    "figsize": (8, 11),
                }
            fig, ax = plt.subplots(nrows, ncols, **fig_kwargs)
        else:
            # set all of ax to None so that each method creates its own ax.
            ax = (
                np.tile(np.asarray(None), nrows * ncols)
                .reshape(nrows, ncols)
                .astype(Axes)
            )
            fig = None
            # call other methods that create their own figures
            self.show_nearest_neighbour()
            self.show_adjacency_matrix("clustermap")
        # first column
        _, ax[0, 0] = self.show_opinions(ax=ax[0, 0])
        _, ax[1, 0] = self.show_adjacency_matrix("mesh", sort=True, ax=ax[1, 0])
        _, ax[2, 0] = self.show_activities(ax=ax[2, 0])
        # second column has opinion as x-axis
        _, ax[0, 1] = self.show_opinions_distribution(ax=ax[0, 1])
        _, ax[1, 1] = self.show_agent_opinions(ax=ax[1, 1], sort=True)
        _, ax[2, 1], cbar = self.show_activity_vs_opinion(ax=ax[2, 1])
        # `show_agent_opinions` already calculates optimal limits
        xlim = ax[1, 1].get_xlim()
        ax[0, 1].set_xlim(*xlim)
        ax[2, 1].set_xlim(*xlim)
        return fig, ax
