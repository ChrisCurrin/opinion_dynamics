import os

from typing import Union

import pandas as pd

from opdynamics.networks import EchoChamber
from opdynamics.utils.constants import DEFAULT_COMPRESSION_LEVEL

_cache_dir = None


def get_cache_dir(sub_path: str = None) -> str:
    """Construct and return a folder for caching under the root 'opinion_dynamics' directory.

    If the `opinion_dynamics` is not in the working tree, the current working directory is used.

    Alternatively, the cache directory to be used can be specified at any point by calling ``set_cache_dir(<path>)``.

    :param sub_path: Optionally specify a subdirectory within the cache directory. Can be of arbitrary depth (limited by
        filesystem, of course).
    :return: Full absolute path to cache directory
    """
    global _cache_dir
    if _cache_dir is None:
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
        local_cache_path = os.path.join(*path, ".cache")
        try:
            os.makedirs(local_cache_path)
        except FileExistsError:
            pass
        _cache_dir = os.path.abspath(local_cache_path)
    if sub_path is not None:
        sub_dir = os.path.join(_cache_dir, sub_path)
        try:
            os.makedirs(sub_dir)
        except FileExistsError:
            pass
        return sub_dir
    return _cache_dir


def set_cache_dir(path: str) -> str:
    """
    Customise where to store the cache by explicitly setting the location.

    Folders will be created if they do not exist.
    :param path: Where to save temporary files.
    :return: Absolute path to directory.
    """
    global _cache_dir
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    _cache_dir = os.path.abspath(path)
    return _cache_dir


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


class NpEncoder(json.JSONEncoder):
    """Numpy Json encoder

    .. code-block :: python

        `json.dumps(data, cls=NpEncoder)`

    see also:
    https://stackoverflow.com/a/57915246

    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
