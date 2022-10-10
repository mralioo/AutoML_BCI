"""
Created on 17.06.21
@author: ali
@Original author : Alexander Lampe
@author: modified by ali
"""
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import IO

from ruamel import yaml as yaml


class LoaderWithCustomTags(yaml.SafeLoader):
    """YAML Loader with `!include` and '!join' constructor.
        Source and license (!include): https://gist.github.com/joshbode/569627ced3076931b02f
        Source (!join): https://stackoverflow.com/a/23212524/3911753

    """

    def __init__(self, stream: IO, version=None, preserve_quotes=None) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream, version=version, preserve_quotes=preserve_quotes)


def _construct_include(loader: LoaderWithCustomTags, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    arg_str = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    args = arg_str.split("::")
    filename = args[0]
    sub_refs = args[1:]

    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            d = yaml.load(f, LoaderWithCustomTags)
            for sub_ref in sub_refs:
                if not isinstance(d, dict):
                    RuntimeError("{:} is not a dict.".format(d))
                else:
                    if sub_ref in d:
                        d = d[sub_ref]
                    else:
                        RuntimeError("Cannot found key {:} in dict {:}", sub_ref, d)
            return d
        else:
            return ''.join(f.readlines())


def _construct_join(loader, node: yaml.Node) -> Any:
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


def make_path_absolute(base_path, path):
    base_path = Path(base_path)
    path = Path(path)
    if path.is_absolute():
        return path
    else:
        return str(base_path / path)


def create_working_folder(root_path, experiment_name, run_name):
    """ Creates a folder structure located in the root path.
    """

    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wdir = os.path.join(root_path, experiment_name, run_name, date_str)

    if os.path.isdir(wdir):
        raise RuntimeWarning("Directory {:} already exists.".format(wdir))
    os.makedirs(wdir, exist_ok=True)

    return wdir


def get_dir_by_indicator(path=sys.path[0], indicator=".git"):
    """ Returns the path of the folder that contains the given indicator

    :param path: Path from where to search the directory tree upwards. (Default value = sys.path[0])
    :type path: str
    :param indicator: Name of the file that indicates the searched directory. (Default value = ".git")
    :type indicator: str
    :raises FileNotFoundError : If any path or any toplevel folder is not found with the given indicator, it raises
    FileNotFoundError.
    """

    is_root = os.path.exists(os.path.join(path, indicator))
    while not is_root:
        new_path = os.path.dirname(path)
        if new_path == path:
            raise FileNotFoundError(
                "Could not find folder containing indicator {:} in any path or any toplevel directory.".format(
                    indicator))
        path = new_path
        is_root = os.path.exists(os.path.join(path, indicator))

    return path


def load_yaml(yaml_path, **kwargs):
    """
    Returns a dict from a yaml file. Enables concatenation of strings with !join [str1, str2, ...]

    :param yaml_path: Path of the yaml file.
    :type yaml_path: str
    :param kwargs: Kwargs accepted from yaml.load function
    """

    if "Loader" not in kwargs:
        kwargs["Loader"] = yaml.SafeLoader
    yaml.add_constructor('!join', _construct_join, kwargs["Loader"])
    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.load(stream, **kwargs)

    return yaml_dict


def dict_place(nested_dict, path, value, construct_missing=True):
    """Place a value in a nested dict following the path to the element and creates dictionary structure if necessary

    Modified from source: https://stackoverflow.com/a/43499625/3911753

    :param nested_dict: A dict of certain depth
    :type nested_dict: dict
    :param path: A list of keys defining the path to the element
    :type path: list of str
    :param value: The value that shall be assigned
    :param construct_missing: Creates nested dict structure on demand, otherwise throws key error.
    :type construct_missing: bool

    """
    pointer = nested_dict

    for key in path[:-1]:
        if key not in pointer:
            if construct_missing:
                pointer[key] = dict()
            else:
                raise KeyError("Key {:} references is invalid. Set construct_missing=True to create missing dicts on "
                               "demand".format(key))
        pointer = pointer[key]
    if (len(path) == 1) and (path[-1] not in nested_dict) and not construct_missing:
        raise KeyError("Key '{:}' is invalid. Set construct_missing=True to create missing dicts on "
                       "demand".format(path[-1]))

    pointer[path[-1]] = value
    return True


def dict_flatten(d, prefix="", recursive=True, sep="."):
    """Flattens a dictionary by concatenating keys of nested dicts with seperator

    :param d: dictionary
    :type d: dict
    :param recursive: Flatten dicts recursively. If recursive is False, this function does nothing except adding the
        prefix to your dictionaries keys. (Default value = True)
    :type recursive: bool
    :param sep: Separator between nested keys (Default value="_")
    :type sep: str
    :returns; Flattened dictionary
    :rtype: dict
    """
    if not isinstance(d, dict):
        raise RuntimeError("Argument d must of type dict, got type {:}.".format(type(d)))
    if not isinstance(prefix, str):
        raise RuntimeError("Argument prefix must of type str, got type {:}.".format(type(prefix)))
    if not isinstance(recursive, bool):
        raise RuntimeError("Argument recursive must of type str, got type {:}.".format(type(recursive)))
    if not isinstance(sep, str):
        raise RuntimeError("Argument sep must of type str, got type {:}.".format(type(sep)))

    fd = dict()
    for key, value in d.items():
        flattened_key = key if prefix == "" else prefix + sep + key
        if isinstance(value, dict) and recursive:
            fd.update(dict_flatten(value, prefix=flattened_key, sep=sep))
        else:
            fd.update({flattened_key: value})
    return fd


def dict_unflatten(flat_dict):
    dd = dict()
    for k, v in flat_dict.items():
        dict_place(dd, k.split("."), v, construct_missing=True)
    return dd


def dict_filter(d: dict, filter_list, return_flattened=False):
    """ Removes key,value paris from a dict of arbitrary structure.
    """
    fd = dict_flatten(d.copy(), sep='.')

    excepts = []
    ignores = []
    global_ignores = []
    for f in filter_list:
        if f[0] == "!":
            excepts.append(f[1:])
        elif f[0] == "*":
            global_ignores.append(f[1:])
        else:
            ignores.append(f)
    keys = fd.keys()
    to_remove = [any([key.startswith(i) for i in ignores]) for key in keys]
    to_remove_global = [any([i in key for i in global_ignores]) for key in keys]
    to_remove = [(rem or rem_glob) for rem, rem_glob in zip(to_remove, to_remove_global)]
    to_keep = [any([i in key for i in excepts]) for key in keys]
    to_remove = [(not k and i) for i, k in zip(to_remove, to_keep)]  # merge conditions

    keys_to_remove = [key for key, rem_flag in zip(keys, to_remove) if rem_flag]

    for key in keys_to_remove:
        del fd[key]

    if return_flattened:
        return fd
    else:
        return dict_unflatten(fd)


def dict_to_yaml(data, yaml_path):
    """
    Returns a yaml file.
    """
    kwargs_dict = {kwarg[0]: kwarg[1] for kwarg in data.items()}
    with open(yaml_path, 'w') as stream:
        yaml.dump(kwargs_dict, stream, default_flow_style=False)


def clean_up_yaml_dict(d):
    d = d.copy()
    for k, v in d.items():
        if isinstance(v, dict):
            if "default" in v and "help" in v:
                d[k] = v["default"]
            else:
                d[k] = clean_up_yaml_dict(v)
    return d


class Tee(object):
    '''
    A Python class like the Unix tee command
    Copies stdout any expcetions to log file
    '''

    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
