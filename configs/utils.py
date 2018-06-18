import yaml
from argparse import Namespace


def dict_to_namespace(dct):
    namespace = Namespace()
    for key, value in dct.items():
        name = key.rstrip("_")
        if isinstance(value, dict) and not key.endswith("_"):
            setattr(namespace, name, dict_to_namespace(value))
        else:
            setattr(namespace, name, value)
    return namespace


def load_config(path: str):
    with open(path) as handler:
        config_data = yaml.load(handler, Loader=yaml.SafeLoader)
        return dict_to_namespace(config_data)

