from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
#from .starcraft2.starcraft2 import StarCraft2Env
import sys
import os
from .aloha import AlohaEnv
from .pursuit import PursuitEnv
from .sensors import SensorEnv
from .hallway import HallwayEnv
from .disperse import DisperseEnv
from .gather import GatherEnv
from .stag_hunt import StagHunt
from .toygame import ToyGameEnv
from .matrixgame import MatrixGameEnv


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
#REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["aloha"] = partial(env_fn, env=AlohaEnv)
REGISTRY["pursuit"] = partial(env_fn, env=PursuitEnv)
REGISTRY["sensor"] = partial(env_fn, env=SensorEnv)
REGISTRY["hallway"] = partial(env_fn, env=HallwayEnv)
REGISTRY["disperse"] = partial(env_fn, env=DisperseEnv)
REGISTRY["gather"] = partial(env_fn, env=GatherEnv)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["toygame"] = partial(env_fn, env=ToyGameEnv)
REGISTRY["matrixgame"] = partial(env_fn, env=MatrixGameEnv)


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
