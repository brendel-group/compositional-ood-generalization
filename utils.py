import argparse
import warnings
from pathlib import Path
from typing import Dict, Union

import yaml


def load_config(path: Union[str, Path]) -> Dict:
    if not isinstance(path, Path):
        path = Path(path)

    with path.open("r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


def save_config(cfg: Dict, path: Union[str, Path]):
    if not isinstance(path, Path):
        path = Path(path)

    with path.open("w") as f:
        yaml.dump(cfg, f)


def kwarg(arg_str: str):
    """Parse key-value pair `key=value` as dict{key: value}"""
    try:
        key, value = arg_str.split("=")

        # custom conversion for common types
        if value == "True":
            value = True
        elif value == "False":
            value = False

        return {key: value}
    except:
        raise argparse.ArgumentTypeError(f"Can't interpret {arg_str} as key-value pair")


class StoreKwargsActions(argparse.Action):
    """Parse list of key-value pairs of the form `key=value` as dict

    Args:
        argparse (_type_): _description_
    """

    def __init__(
        self, option_strings, dest, nargs="+", default=None, type=kwarg, **kwargs
    ):
        if nargs != "+":
            raise ValueError("kwargs action only allows `nargs=+`")
        if type is not kwarg:
            type = kwarg
            warnings.warn(
                "kwargs action cannot do type conversion, type will be ignored"
            )
        if not (isinstance(default, dict) or default is None):
            raise ValueError("kwargs action default must be `dict` or `None`")
        super().__init__(option_strings, dest, nargs, default, type, **kwargs)

    def __call__(self, parser, namespace, values, optin_string=None):
        # NOTE contrary to the argparse docs, values do not have the type conversion applied
        #  we therefore have to apply it here manually
        d = {}
        for item in values:
            d.update(kwarg(item))
        setattr(namespace, self.dest, d)
