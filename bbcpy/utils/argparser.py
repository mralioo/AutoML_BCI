"""
Created on 31.08.21
@Original author : Alexander Lampe
@author: modified by ali
"""
import argparse
import os
from pathlib import Path

from ruamel import yaml as yaml

from bbcpy.utils.file_management import load_yaml, LoaderWithCustomTags, make_path_absolute, clean_up_yaml_dict, \
    dict_filter


class YamlFileActions(argparse.Action):
    """Action to parse a yaml file and load it's parameters"""

    def __call__(self, parser, args, path, option_string=None):
        yaml_path, subref = parse_yaml_path(path)

        if not Path(yaml_path).is_file():
            raise RuntimeError("Error using {:} {:}. File  not found.".format(option_string, yaml_path))
        yaml_dict = load_yaml_with_imports(path)
        if not isinstance(yaml_dict, dict):
            raise RuntimeError("Yaml file {:} must define a dictionary of command line arguments"
                               ", got type {:}.".format(yaml_path, type(yaml_dict)))

        yaml_dict = clean_up_yaml_dict(yaml_dict)
        # yaml_args = flatten_dicts(yaml_dict)

        setattr(args, self.dest, yaml_path)

        for key, value in yaml_dict.items():
            setattr(args, key, value)


class MultiKeyValuePairsAction(argparse.Action):
    """Action to parse multiple whitespace separated key-value-pairs defined with 'key1=value1 key2=value2 ...' and
    store them in a dictionary"""

    def __call__(self, parser, args, values, option_string=None):
        for kv in values:
            try:
                (k, v) = kv.split("=", 2)
            except ValueError as e:
                raise argparse.ArgumentError(self, f"could not parse argument \"{values[0]}\" as k=v format")
            d = getattr(args, self.dest) or {}
            d[k] = v
            setattr(args, self.dest, d)


def args_to_yaml(args, yaml_path):
    """
    Returns a yaml file.
    """
    kwargs_dict = {kwarg[0]: kwarg[1] for kwarg in args._get_kwargs()}
    with open(yaml_path, 'w') as stream:
        yaml.dump(kwargs_dict, stream)


def args_from_dict(d):
    """Creates (nested) argparse.Namespace objects from (nested) dict.

        :param d: a dict
        :type d: dict
        :returns: (nested) argparse.Namespace object
        :rtype: argparse.Namepsace
        """

    args = argparse.Namespace(**d)
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(args, k, args_from_dict(v))
    return args


def args_to_dict(args):
    """Creates a dict from (nested) argparse.Namespace objects
    """

    d = vars(args).copy()
    for k, v in d.items():
        if isinstance(v, argparse.Namespace):
            d[k] = args_to_dict(v)

    return d


# def yaml_argparse(yaml_path, raw_args=None):
#     """
#     Returns hyperparameters dict as a ArgumentParser object .
#
#     This function first needs to have a template yaml file, where all needed hyperparameters set is defined.
#     Then according to the given argument, the function sets a new value of an existing hyperparameter key
#     or add a new one.
#     These hyperparameters are stored as Argumentparser object and can be accessed directly with the dot notation.
#
#     Also, you can add yaml file, that contained customized configuration of hyperparameters, as an argument.
#     The new values will be overridden, otherwise the default values wil remain.
#
#     :param yaml_path: The path of the default hyperparameter yaml file.
#     :param raw_args:  Sets value of an exiting hyperparameter argument e.g raw_args=['--key', 'value' ]
#                      (Default value = None)
#
#     """
#     parser = argparse.ArgumentParser(description=__doc__,
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("-f", "--file",
#                         dest="yaml_file",
#                         help="yaml file containing experiment parameters. Stored as attribute 'yaml_file' in parsed "
#                              "args.",
#                         metavar="FILE",
#                         required=False)
#
#     class ListAction(argparse.Action):
#         """Creates list from string containing a list expression"""
#
#         def __call__(self, parser, namespace, values, option_string=None):
#             setattr(namespace, self.dest, eval(values))
#
#     hparam_dict = load_yaml(yaml_path)
#
#     for key, value in hparam_dict.items():
#         # kwargs unpacking
#         if isinstance(value, dict):
#             kwargs = dict(value)
#             value = kwargs.pop("default")
#         else:
#             kwargs = {}
#
#         if isinstance(value, list):
#             parser.add_argument("--" + key, action=ListAction, default=value, **kwargs)
#         else:
#             parser.add_argument("--" + key, type=type(value), default=value, **kwargs)
#
#     def _get_cmd_args(args):
#         cmd_args = []
#         [cmd_args.append(x[2::]) for x in args if x[0:2] == "--"]
#         return cmd_args
#
#     # get command line arguments to prevent overriding them when applying parameters from another yaml file
#     if raw_args:
#         raw_args_cmd = _get_cmd_args(raw_args)
#     elif len(sys.argv) > 1:
#         raw_args_cmd = _get_cmd_args(sys.argv[1::])
#     else:
#         raw_args_cmd = []
#
#     args = parser.parse_args(raw_args)
#     if args.yaml_file:
#         hparam_dict_yaml = load_yaml(args.yaml_file)
#         # hparam analysis
#         [print("Obsolete parameter '{:}' in {:}".format(key, args.yaml_file)) for key in hparam_dict_yaml.keys()
#          if (key not in hparam_dict and not key == "yaml_file")]
#         [print("Missing parameter {:} in {:}. Using default value.".format(key, args.yaml_file)) for key in
#          hparam_dict.keys() if key not in hparam_dict_yaml]
#
#         for key, value in hparam_dict.items():
#             value = value["default"] if isinstance(value, dict) else value
#             if (key in hparam_dict_yaml) and (key not in raw_args_cmd):
#                 if isinstance(hparam_dict_yaml[key], dict):
#                     if "default" in hparam_dict_yaml[key]:
#                         setattr(args, key, hparam_dict_yaml[key]["default"])
#                     else:
#                         setattr(args, key, hparam_dict_yaml[key])
#                 else:
#                     setattr(args, key, hparam_dict_yaml[key])
#     else:
#         args.yaml_file = str(yaml_path)
#
#     return args


def add_yaml_file_parser(parser):
    parser.add_argument("-f", "--file",
                        help="""Path to yaml file. You can sub reference the content with '::', e.g. --file 
                             path/to/file.yaml::key.""",
                        metavar="FILE",
                        required=False,
                        action=YamlFileActions)


def add_tag_parser(parser):
    parser.add_argument("--tags",
                        nargs='*',
                        action=MultiKeyValuePairsAction,
                        metavar="KEY=VALUE",
                        help="Set one or multiple key-value pairs (key=value) that are logged as tags with MlFlow and "
                             "as parameters with tensorboard. Do not put spaces before or after the = sign."
                             " If a value contains spaces, you should define it with double quotes: 'foo=this is a "
                             "sentence'. Note that values are always treated as strings.")


def get_argument_parser():
    """Command line argument parser with required argument --file to define a parameter
    yaml file and optional argument --tag to define custom tags that are logged as parameters in tensorboard and tags
    in mlflow.
    The parser supports imports within a yaml file using the 'imports' key (the key is reserved):

        .. code-block:: yaml

            imports:
                - path/to/yaml

    """

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_yaml_file_parser(parser)
    add_tag_parser(parser)
    return parser


def parse_yaml_path(yaml_path):
    parsed_path = str(yaml_path).split("::")
    p = parsed_path[0]
    sub_refs = parsed_path[1:] if len(parsed_path) > 1 else []
    return p, sub_refs


def load_args_from_yaml(path):
    d = load_yaml_with_imports(path)
    d = clean_up_yaml_dict(d)
    d = dict_filter(d, filter_list=["*imports"], return_flattened=False)
    args = args_from_dict(d)
    return args


def update_args(args, param_dict):
    """Update argparse.Namespace object with parameter dict

    param_dict must match the data structure of args. Attributes of args are replaced with param_dict[attrib_name]
    if type is not dict. If type of an attribute of args is dict, it is updated with param_dict[attrib_name].

    :param args: args obtained with parser.parse_args()
    :param param_dict: A dict with parameters
    :type param_dict: dict
    """

    if isinstance(param_dict, dict):
        for k, v in param_dict.items():
            if hasattr(args, k):
                if isinstance(getattr(args, k), dict):
                    getattr(args, k).update(v)
                elif isinstance(getattr(args, k), argparse.Namespace):
                    update_args(getattr(args, k), v)
                else:
                    if isinstance(v, dict):
                        setattr(args, k, args_from_dict(v))
                    else:
                        setattr(args, k, v)
            else:
                if isinstance(v, dict):
                    setattr(args, k, args_from_dict(v))
                else:
                    setattr(args, k, v)
    return args


def load_yaml_with_imports(path):
    import_key = "imports"
    path, sub_refs = parse_yaml_path(path)
    yaml_dict = load_yaml(path, Loader=LoaderWithCustomTags)

    for sub_ref in sub_refs:
        if sub_ref not in yaml_dict:
            raise RuntimeError("Cannot find subref {:} in loaded yaml file {:}".format(sub_ref, yaml_dict))
        yaml_dict = yaml_dict[sub_ref]

    def _iter_dict(d, base_path):
        nd = dict()
        for key, value in d.items():

            if key == import_key:
                if not isinstance(value, list) and not isinstance(value, tuple):
                    raise RuntimeError("Wrong data type of attribute {:}. Expected list or tuple, got {:}".format(
                        import_key, type(value)))

                for item in value:
                    nd.update(load_yaml_with_imports(make_path_absolute(base_path, item)))
                    nd.update(d)
            elif isinstance(value, dict):
                nd[key] = _iter_dict(value.copy(), base_path)

        d.update(nd)
        return d

    return _iter_dict(yaml_dict, os.path.dirname(path))


def yaml_argparse(yaml_path, raw_args=None):
    """Returns hyperparameters dict as a ArgumentParser object .
    """

    hparam_dict = load_yaml_with_imports(yaml_path)
    hparam_dict = clean_up_yaml_dict(hparam_dict)
    parsed_args = args_from_dict(hparam_dict)
    args = update_args(parsed_args, raw_args)

    return args


