from robo_transformers.models.rt1.agent import RT1Agent
from robo_transformers.models.octo.agent import OctoAgent
from robo_transformers.models.teleop import TeleOpAgent
from robo_transformers.models.replay import ReplayAgent
REGISTRY = {
    "rt1": {
        "agent": RT1Agent,
        "variants": ["rt1main", "rt1simreal", "rt1multirobot", "rt1x"]
    },
    "octo": {
        "agent": OctoAgent,
        "variants": ["octo-small", "octo-base"]
    },
    "teleop": {
        "agent":  TeleOpAgent,
       
    },
    "replay": {
        "agent": ReplayAgent,
    },

}
