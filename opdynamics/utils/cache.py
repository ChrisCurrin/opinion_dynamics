import os

from typing import Union

import pandas as pd

from opdynamics.networks.echochamber import EchoChamber
from opdynamics.utils.constants import DEFAULT_COMPRESSION_LEVEL


def get_cache_dir() -> str:
    """Construct a folder for caching under the root 'opinion_dynamics' directory.

    If the `opinion_dynamics` is not in the working tree, the current working directory is used."""
    pwd = os.getcwd()
    path = os.path.split(pwd)
    while (
        path[0] != ""
        and path[-1] != ""
        and "opinion_dynamics" not in path[-1]
        and path[-1] != "content"
    ):
        path = os.path.split(path[0])
    if path[0] == "" or path[-1] == "":
        path = pwd
    try:
        os.makedirs(os.path.join(*path, ".cache"))
    except FileExistsError:
        pass
    return os.path.join(*path, ".cache")


def cache_ec(cache: Union[str, int, bool], ec: EchoChamber, write_mapping=True) -> None:
    """ Cache an echochamber object if `cache` is neither `False` nor `None`.

    :param cache: Either "all" or True (last time point) or ℤ ∈ [1, 9] for compression level (last time point).
    :param ec: EchoChamber (or subclass) object to save.
    :param write_mapping: Write to a file that maps the object's string representation and it's hash value.

    """
    if cache:
        if type(cache) is str and "all" in cache:
            cache = cache.replace("all", "")
            complevel = int(cache) if len(cache) else DEFAULT_COMPRESSION_LEVEL
            ec.save(only_last=False, complevel=complevel, write_mapping=write_mapping)
        else:
            complevel = cache if cache > 1 else DEFAULT_COMPRESSION_LEVEL
            ec.save(only_last=True, complevel=complevel, write_mapping=write_mapping)


def save_results(file_name: str, ec: EchoChamber, **kwargs) -> None:
    """ Save ``EchoChamber`` agents' opinions to a shared HDF5 DataFrame.

    :param file_name: Full path to results file (*file* - not directory - created if it does not exist).
    :param ec: EchoChamber after a `run_network` operation.
    :param kwargs: Full list of keyword arguments used.
    :return:
    """
    df_builder = []

    kwargs.pop("cls", None)
    kwargs.pop("method", None)

    # put data into dictionaries with keys for column names
    for y_idx, opinion in enumerate(ec.result.y[:, -1]):
        df_builder.append({"i": y_idx, "opinion": opinion, **kwargs})
    with pd.HDFStore(file_name) as store:
        store.append("df", pd.DataFrame(df_builder))
