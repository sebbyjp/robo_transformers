from robo_transformers.models.rt1.action import RT1Action
from robo_transformers.models.rt1.agent import RT1Agent
from robo_transformers.models.octo.action import OctoAction
from robo_transformers.models.octo.agent import OctoAgent
from robo_transformers.models.teleop import TeleOpAgent

REGISTRY = {
    "rt1": {
        "agent": RT1Agent,
        "action": RT1Action,
        "weight_keys": ["rt1main", "rt1simreal", "rt1multirobot", "rt1x"]
    },
    "octo": {
        "agent": OctoAgent,
        "action": OctoAction,
        "weight_keys": ["octo-small", "octo-base"]
    },
    "teleop": {
        "agent": TeleOpAgent,
        "action": RT1Action,
        "weight_keys": []
    }
}
