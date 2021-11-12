"""
Run a large parameter range to plot noise (D) x sample_size (n) x <another variable>
# todo: create click script
"""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.random.seed(1337)

sns.set(
    context="notebook",
    style="ticks",
    rc={
        "pdf.fonttype": 42,  # embed font in output
        "figure.facecolor": "white",
        "axes.facecolor": "None",
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
    },
)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("asyncio").setLevel(logging.WARNING)

if __name__ == "__main__":
    from opdynamics.simulation import run_product
    from opdynamics.socialnetworks import SampleChamber
    from opdynamics.utils.cache import set_cache_dir
    from opdynamics.utils.logging import LoggingContext
    from opdynamics.visualise.dense import plot_surfaces

    cache_dir = ".cache"
    old_cache_dir, new_dir = set_cache_dir(cache_dir)

    parameters = dict(
        N=1000,
        m=10,
        T=5.0,
        epsilon=1e-2,
        gamma=2.1,
        dt=0.01,
        r=0.5,
        update_conn=True,
        cls=SampleChamber,
        method="RK45",
        name="sample chamber",
    )

    D_range = np.round(np.arange(0.0, 5.0001, 1), 3)

    if len(sys.argv) > 1:
        sample_size_range = [int(sys.argv[-1])]
    else:
        sample_size_range = [1, 30, 50]

    range_parameters = {
        "D": {"range": D_range, "title": "D"},
        "sample_size": {"range": sample_size_range, "title": "n"},
        "alpha": {"range": [1, 2, 3], "title": "α"},
        "beta": {"range": [1, 2, 3], "title": "β"},
        "K": {"range": [1, 2, 3], "title": "K"},
        "background": {"range": [True]},
        "sample_method": {
            "range": ["full", "subsample", "simple"],
            "title": "sample method",
        },
        "seed": [
            1337,
            # 10101,
            # 12345
        ],
    }

    with LoggingContext(logging.WARNING) as logger:
        with LoggingContext(logging.WARNING, logger="SocialNetwork"):
            with LoggingContext(logging.WARNING, logger="cache"):
                df = run_product(
                    range_parameters,
                    noise_start=0,
                    cache=True,
                    cache_sim="nudge_range",
                    cache_mem=False,
                    parallel=True,
                    **parameters,
                )
        logger.info(f"simulations completed")

    if len(sample_size_range) > 1 and isinstance(df, pd.DataFrame):
        # plot range of sample sizes

        plot_surfaces(df, "D", "sample_size", parameters, range_parameters)
        plt.show()
