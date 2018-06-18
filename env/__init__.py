from .gathering_env import GatheringEnv
from .maps import get_map_path

ALL_ENV = [
    "GatheringEnv"
]


def get_env(name, cfg):
    # @name         : name of the environment
    # @cfg          : configuration namespace

    assert name in ALL_ENV, "ENV %s is not on defined." % name

    _Env = None

    if name == "GatheringEnv":
        _Env = GatheringEnv

    return _Env(cfg)
