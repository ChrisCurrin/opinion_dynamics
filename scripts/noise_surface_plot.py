"""
Run a large parameter range to plot noise (D) x sample_size (n) x <another variable>
"""

if __name__ == "__main__":
    import sys

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from opdynamics.networks.echochamber import SampleChamber
    from opdynamics.simulation import run_product
    from opdynamics.visualise.dense import plot_surfaces

    parameters = dict(
        N=1000,
        m=10,
        T=10.0,
        epsilon=1e-2,
        gamma=2.1,
        dt=0.01,
        K=3,
        beta=3,
        alpha=3,
        r=0.5,
        cls=SampleChamber,
        method="Euler",
    )

    D_range = np.round(np.arange(0.0, 5.0001, 0.1), 3)

    if len(sys.argv) > 1:
        sample_size_range = [int(sys.argv[-1])]
    else:
        sample_size_range = np.arange(1, 50.0001, 1, dtype=int)

    range_variables = {
        "D": {"range": D_range, "title": "D"},
        "sample_size": {"range": sample_size_range, "title": "n"},
        "alpha": {"range": [1, 2, 3], "title": "α"},
        "beta": {"range": [0, 1, 2, 3], "title": "β"},
        "K": {"range": [1, 2, 3]},
    }

    df = run_product(
        range_variables,
        noise_start=0,
        cache=True,
        cache_sim=True,
        parallel=True,
        **parameters
    )

    if len(sample_size_range) > 1 and isinstance(df, pd.DataFrame):
        # plot range of sample sizes

        plot_surfaces(df, "D", "sample_size", parameters, range_variables)
        plt.show()
