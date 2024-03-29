import re
from pathlib import Path
from typing import Dict, List, Any, TypeVar

import yaml


def get_digit_subscript(i: int):
    assert i in range(10), f"Subscripts are only available for digits 0-9, but got {i}."
    return chr(0x2080 + i)


def all_equal(l: List):
    return l.count(l[0]) == len(l)


def load_config(path: Path) -> Dict:
    # fix for scientific notation from
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    with path.open("r") as f:
        try:
            return yaml.load(f, Loader=loader)
        except yaml.YAMLError as exc:
            print(exc)


# from pydantic library
# https://github.com/pydantic/pydantic/blob/9d631a3429a66f30742c1a52c94ac18ec6ba848d/pydantic/utils.py#L198
KeyType = TypeVar("KeyType")


def deep_update(
    mapping: Dict[KeyType, Any], *updating_mappings: Dict[KeyType, Any]
) -> Dict[KeyType, Any]:
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping
