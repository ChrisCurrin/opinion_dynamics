"""
Run a large parameter range to plot noise (D) x sample_size (n) x <another variable>
"""
import itertools
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from opdynamics.dynamics.echochamber import SampleChamber
from opdynamics.metrics.opinions import distribution_modality
from opdynamics.simulation import run_product

parameters = dict(
    N=1000,
    m=10,
    T=10,
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
    range_variables, noise_start=0, cache=True, cache_sim=True, **parameters
)

if len(sample_size_range) > 1:
    # plot range of sample sizes

    del parameters["cls"]
    del parameters["method"]

    def mask_df(masks):
        N_val = masks.pop("N", 1000)
        mask = df["N"] == N_val
        for key, value in masks.items():
            mask = np.logical_and(mask, df[key] == value)
        return df[mask]

    def plot_surfaces(params, variables):
        z_vars = [k for k in variables if k != "D" and k != "sample_size"]
        fig, axs = plt.subplots(nrows=len(z_vars))

        for ax, key in zip(axs, z_vars):
            default_kwargs = {k: v for k, v in params.items() if k != key}
            df = mask_df(default_kwargs)
            z = pd.DataFrame(columns=sample_size_range, index=D_range, dtype=np.float64)
            for D, sample_size in itertools.product(D_range, sample_size_range):
                mask = np.logical_and(df["D"] == D, df["sample_size"] == sample_size)
                z.loc[D, sample_size] = distribution_modality(df.loc[mask, "opinion"])
            mesh = ax.pcolormesh(D_range, sample_size_range, z, cmap="viridis")
            cbar = fig.colorbar(mesh, ax=ax, cmap="viridis")
            desc = variables[key]["title"] if "title" in variables[key] else key
            ax.set_xlabel("D")
            ax.set_ylabel("n")
            cbar.ax.set_title(desc)

    plot_surfaces(parameters, range_variables)
    plt.show()
