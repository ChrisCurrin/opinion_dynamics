import os

from typing import Tuple, Union

import logging
import json
import numpy as np
import pandas as pd

from opdynamics.socialnetworks import SocialNetwork
from opdynamics.utils.constants import DEFAULT_COMPRESSION_LEVEL


logger = logging.getLogger("cache")

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


def set_cache_dir(path: str) -> Tuple[str, str]:
    """
    Customise where to store the cache by explicitly setting the location.

    Folders will be created if they do not exist.
    :param path: Where to save temporary files.
    :return: Previous path, new path.
    """
    global _cache_dir
    if _cache_dir:
        old_dir = _cache_dir
    else:
        old_dir = get_cache_dir()
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    _cache_dir = os.path.abspath(path)
    return old_dir, _cache_dir


class cache_dir:
    """Context manager for changing the current cache directory"""

    def __init__(self, new_path):
        try:
            os.makedirs(new_path)
        except FileExistsError:
            pass
        self.new_path = os.path.abspath(new_path)

    def __enter__(self):
        global _cache_dir
        self.saved_dir, _cache_dir = _cache_dir, self.new_path

    def __exit__(self, etype, value, traceback):
        global _cache_dir
        _cache_dir = self.saved_dir


def cache_ec(
    cache: Union[str, int, bool], sn: SocialNetwork, write_mapping=True
) -> None:
    """Cache an SocialNetwork object if `cache` is neither `False` nor `None`.

    :param cache: Either "all" or True (last time point) or ℤ ∈ [1, 9] for compression level (last time point).
    :param sn: SocialNetwork (or subclass) object to save.
    :param write_mapping: Write to a file that maps the object's string representation and it's hash value.

    """
    if cache:
        if type(cache) is str:
            cache = cache.replace("_", "").replace(" ", "").lower()
            only_last = "all" not in cache
            cache_interactions = "opinion" in cache
            cache = cache.replace("all", "")
            cache = cache.replace("opinion", "")
        
            complevel = int(cache) if cache.isnumeric() else DEFAULT_COMPRESSION_LEVEL
            sn.save(
                only_last=only_last,
                interactions=cache_interactions,
                complevel=complevel,
                write_mapping=write_mapping,
            )
        else:
            complevel = cache if cache > 1 else DEFAULT_COMPRESSION_LEVEL
            sn.save(only_last=True, complevel=complevel, write_mapping=write_mapping)


def save_results(file_name: str, sn: SocialNetwork, **kwargs) -> None:
    """Save ``SocialNetwork`` agents' opinions to a shared HDF5 DataFrame.

    :param file_name: Full path to results file (*file* - not directory - created if it does not exist).
    :param sn: SocialNetwork after a `run_network` operation.
    :param kwargs: Full list of keyword arguments used.
    :return:
    """
    df_builder = []

    cls = kwargs.pop("cls", None)
    kwargs["cls"] = cls.__name__ if cls is not None else ""

    method = kwargs.pop("method", None)
    kwargs["method"] = method if method is not None else ""

    activity_distribution = kwargs.pop("activity_distribution", None)
    kwargs["activity_distribution"] = (
        activity_distribution.__name__ if activity_distribution is not None else ""
    )

    kwargs.pop("plot_opinion", None)

    kwargs["name"] = sn.name

    # put data into dictionaries with keys for column names
    for y_idx, opinion in enumerate(sn.result.y[:, -1]):
        df_builder.append({"i": y_idx, "opinion": opinion, **kwargs})

    with pd.HDFStore(file_name) as store:
        try:
            df = pd.DataFrame(df_builder)
            min_itemsize = {
                "name": 100,
                "cls": 30,
                "activity_distribution": 30,
            }
            if "sample_method" in df.columns:
                min_itemsize["sample_method"] = 30

            store.append(
                "df",
                df,
                format="table",
                data_columns=True,
                index=False,
                min_itemsize=min_itemsize,
            )
        except ValueError as err:
            logger.error(f"Could not save results to {file_name}")
            logger.debug(f"kwargs = {kwargs}")
            current_columns = store["df"].columns
            new_columns = pd.DataFrame(df_builder).columns
            logger.debug(
                f"new (first line) vs old(second line)\n'{new_columns}'\n'{current_columns}'"
            )
            raise err


def get_hash_filename(hashable_obj, filetype="h5", extra="", cache_kwargs=None) -> str:
    """Get a cacheable filename for this class instance (must be decorated by `utils.decorators.hashable`)"""

    cache_kwargs = cache_kwargs or {}

    if filetype.startswith("."):
        filetype = filetype[1:]

    cache_dir = get_cache_dir(**cache_kwargs)

    return os.path.join(cache_dir, f"{hashable_obj.hash_extra(extra)}.{filetype}")


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
