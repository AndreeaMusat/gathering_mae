from .gathering_env import GatheringEnv
from .single_agent_wrapper import SingleAgentGatheringEnv

ALL_ENV = [
    "GatheringEnv", "SingleAgentGatheringEnv"
]


def get_env(name, cfg):
    # @name         : name of the environment
    # @cfg          : configuration namespace

    assert name in ALL_ENV, "ENV %s is not on defined." % name

    _Env = None

    if name == "GatheringEnv":
        _Env = GatheringEnv
    elif name == "SingleAgentGatheringEnv":
        _Env = SingleAgentGatheringEnv

    return _Env(cfg)
