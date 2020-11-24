import logging
import os
import opdynamics.visualise.compat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
from matplotlib.collections import QuadMesh
from matplotlib.colors import LogNorm, colorConverter

from opdynamics.networks import EchoChamber
from opdynamics.utils.decorators import optional_fig_ax
from opdynamics.utils.distributions import negpowerlaw
from opdynamics.utils.plot_utils import get_equal_limits

logger = logging.getLogger("animate")


class Animator(object):
    def __init__(self, ec: EchoChamber, vis_kwargs=None):
        from opdynamics.visualise import VisEchoChamber

        self.ec = ec
        self.vis = VisEchoChamber(ec)
        self.vis_kwargs = vis_kwargs or {}
        self.animations = {}

    @optional_fig_ax
    def animate_opinions(self, fig=None, ax=None):
        def init():
            self.vis.show_opinions_snapshot(ax=ax, t=0, **self.vis_kwargs)
            ax.set_xlim(*get_equal_limits(self.ec.result.y))
            ax.set_ylim(0, 1)
            ax.set_title(f"{0:>6.3f}")

        def animate(i):
            ax.clear()
            self.vis.show_opinions_snapshot(ax=ax, t=i, **self.vis_kwargs)
            ax.set_title(f"{self.ec.result.t[i]:>6.3f}")

        if len(fig.axes) > 1:
            self.animations["opinions"] = fig, animate, init, False
        else:
            self.animations["opinions"] = animation.FuncAnimation(
                fig,
                animate,
                init_func=init,
                frames=len(self.ec.result.t),
                repeat=False,
            )

    @optional_fig_ax
    def animate_social_interactions(self, fig=None, ax=None):
        ax.set_xlim(0, self.ec.N)
        ax.set_ylim(0, self.ec.N)
        norm = LogNorm(1, np.max(self.ec.adj_mat.accumulator))
        cmap = "magma_r"

        accumulate = self.ec.adj_mat.accumulate
        total_interactions = pd.DataFrame(
            self.ec.adj_mat.accumulator,
            columns=pd.Index(np.arange(self.ec.N), name="i"),
            index=pd.Index(np.arange(self.ec.N), name="j"),
        )
        sorted_df = total_interactions.sort_values(
            by=list(total_interactions.index), axis="index"
        ).sort_values(by=list(total_interactions.columns), axis="columns")

        mesh: QuadMesh = ax.pcolormesh(accumulate(0), norm=norm, cmap=cmap)
        fig.colorbar(mesh, ax=ax)
        ax.set_title(f"{0:>6.2f}")

        def animate(i):
            # assign new values to dataframe
            total_interactions.iloc[:] = accumulate(i)
            # apply sorting, get values, and ravel() to flatten array (as expected by `set_array`)
            mesh.set_array(
                total_interactions.loc[
                    sorted_df.index, sorted_df.columns
                ].values.ravel()
            )
            ax.set_title(f"{self.ec.result.t[i]:>6.2f}")
            return (mesh,)

        # if part of a subplot (>2 because cbar_ax is added in call to fig.colorbar)
        if len(fig.axes) > 2:
            self.animations["interactions"] = fig, animate, None, True
        else:
            self.animations["interactions"] = animation.FuncAnimation(
                fig,
                animate,
                frames=self.ec.adj_mat._time_mat.shape[0],
                repeat=False,
                blit=True,
            )

    def animate_nearest_neighbour(self, **kwargs):
        # TODO: use custom jointplot
        bw_adjust = kwargs.setdefault("bw_adjust", 0.5)
        g = self.vis.show_nearest_neighbour(**kwargs)
        xlim = g.ax_joint.get_xlim()
        ylim = g.ax_joint.get_ylim()
        color = kwargs.pop("color", "Purple")
        color_rgb = colorConverter.to_rgb(color)
        colors = [
            sns.set_hls_values(color_rgb, l=l) for l in np.linspace(1, 0, 12)  # noqa
        ]
        # Make a colormap based off the plot color
        cmap = sns.blend_palette(colors, as_cmap=True)
        kwargs.setdefault("cmap", cmap)
        kwargs.setdefault("shade", True)
        marginal_kws = kwargs.pop("marginal_kws", dict())
        marginal_kws.update(bw_adjust=bw_adjust)
        marginal_kws.setdefault("color", color)
        marginal_kws.setdefault("shade", True)

        def init():
            g.ax_joint.clear()
            g.ax_marg_x.clear()
            g.ax_marg_y.clear()
            g.fig.suptitle(f"{0:>6.3f}")

        def animate(i):
            g.ax_joint.clear()
            g.ax_marg_x.clear()
            g.ax_marg_y.clear()
            x = self.ec.result.y[:, i]
            y = self.ec.get_nearest_neighbours(t=i)
            # drop nans
            not_na = pd.notnull(x) & pd.notnull(y)
            g.x = x[not_na]
            g.y = y[not_na]
            g.plot_joint(sns.kdeplot, **kwargs)
            g.plot_marginals(sns.kdeplot, **marginal_kws)

            # these are reset after .clear(); so go correct these as in joint_plot
            plt.setp(g.ax_marg_x.get_xticklabels(), visible=False)
            plt.setp(g.ax_marg_y.get_yticklabels(), visible=False)
            plt.setp(g.ax_marg_x.yaxis.get_majorticklines(), visible=False)
            plt.setp(g.ax_marg_x.yaxis.get_minorticklines(), visible=False)
            plt.setp(g.ax_marg_y.xaxis.get_majorticklines(), visible=False)
            plt.setp(g.ax_marg_y.xaxis.get_minorticklines(), visible=False)
            plt.setp(g.ax_marg_x.get_yticklabels(), visible=False)
            plt.setp(g.ax_marg_y.get_xticklabels(), visible=False)
            g.fig.suptitle(f"{self.ec.result.t[i]:>6.3f}", va="bottom")

        self.animations["nearest_neighbour"] = animation.FuncAnimation(
            g.fig,
            animate,
            init_func=init,
            frames=len(self.ec.result.t),
            repeat=False,
        )

    def run_coupled_animations(self, frames=-1, methods=None):
        if frames == -1:
            frames = len(self.ec.result.t)

        # go through each method and populate self.animations with name:(fig, animate, init, blit)
        if methods is not None:
            fig, ax = plt.subplots(1, len(methods))
            for meth, m_ax in zip(methods, ax):
                meth(fig=fig, ax=m_ax)
        else:
            if len(self.animations) == 0:
                raise RuntimeError(
                    "animation methods need to be specified or called beforehand."
                )
            else:
                fig = plt.gcf()
        # collect all animations' init and animate methods
        init_methods = []
        animate_methods = []
        figs = []
        blit = True

        for name, anim in self.animations.items():
            _fig, animate, init, _blit = anim
            figs.append(_fig)
            animate_methods.append(animate)
            if init is not None:
                init_methods.append(init)
            blit = blit and _blit

        def init():
            """define parent init to call"""
            for _init_method in init_methods:
                _init_method()

        def animate(i):
            """define parent animate to call"""
            objs = []
            for _animate_method in animate_methods:
                objs.append(_animate_method(i))
            return objs

        # run animation
        animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=frames,
            repeat=False,
            blit=blit,
        )

    def save(self):
        logger.info("saving...")
        try:
            os.makedirs("output")
        except FileExistsError:
            pass
        for name, anim in self.animations.items():
            logger.debug(f"\t {name}")
            anim.save(os.path.join("output", f"{name}.mp4"))
        logger.info("saved")


if __name__ == "__main__":
    import opdynamics.simulation as Simulation

    logging.basicConfig(level=logging.DEBUG)

    _kwargs = dict(
        N=1000,  # number of agents
        m=10,  # number of other agents to interact with
        alpha=3,  # controversialness of issue (sigmoidal shape)
        K=3,  # social interaction strength
        epsilon=1e-2,  # minimum activity level with another agent
        gamma=2.1,  # power law distribution param
        beta=3,  # power law decay of connection probability
        activity_distribution=negpowerlaw,
        r=0.5,
        dt=0.01,
        T=5,
    )

    ec = Simulation.run_params(
        EchoChamber, plot_opinions=False, lazy=False, cache=False, **_kwargs
    )
    vis = VisEchoChamber(ec)
    vis.show_opinions()
    animator = Animator(ec)
    # _fig, _ax = plt.subplots(1, 2)
    # animator.animate_social_interactions()
    animator.animate_opinions()
    animator.animate_nearest_neighbour()
    animator.save()
    plt.show()
