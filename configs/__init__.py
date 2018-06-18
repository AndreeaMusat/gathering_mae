from .utils import load_config

ALL_CONFIGS = [
    "default"
]


def get_config(name):
    # @name         : name of the config

    assert name in ALL_CONFIGS, "Config %s is not defined." % name

    return load_config(f"configs/{name}.yaml")
