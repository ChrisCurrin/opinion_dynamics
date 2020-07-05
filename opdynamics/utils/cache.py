import os


def get_cache_dir() -> str:
    """Construct a folder for caching under the root 'opinion_dynamics' directory.

    If the `opinion_dynamics` is not in the working tree, the current working directory is used."""
    pwd = os.getcwd()
    path = os.path.split(pwd)
    while path[0] != "" and path[-1] != "" and (path[-1] != "opinion_dynamics" or path[-1] != "content"):
        path = os.path.split(path[0])
    if path[0] == "" or path[-1] == "":
        path = pwd
    try:
        os.makedirs(os.path.join(*path, ".cache"))
    except FileExistsError:
        pass
    return os.path.join(*path, ".cache")
