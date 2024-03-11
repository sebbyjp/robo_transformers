from robo_transformers.models.rt1.agent import RT1Agent
from robo_transformers.models.octo.agent import OctoAgent
from robo_transformers.models.teleop import TeleOpAgent
from robo_transformers.models.replay import ReplayAgent

from robo_transformers.interface import Agent
REGISTRY = {
    "gs": {
        "rt1": RT1Agent,
    },
    "hf": {
        "rail-berkley": OctoAgent,

    },
    "mbd": {
        "teleop":  TeleOpAgent,
        "replay": ReplayAgent,
    },
}
