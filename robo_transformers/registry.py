from robo_transformers.models.rt1.action import RT1Action
from robo_transformers.models.rt1.agent import RT1Agent
from robo_transformers.models.octo.action import OctoAction
from robo_transformers.models.octo.agent import OctoAgent
from robo_transformers.models.teleop import TeleOpAgent
# from robo_transformers.models.rtdiffusion.agent import RTDiffusion
from robo_transformers.spaces.common import EEF_ACTION_SPACE
from robo_transformers.spaces.common import BASIC_VISION_LANGUAGE_OBSERVATION_SPACE as VLA_SPACE

REGISTRY = {
    "rt1": {
        "agent": RT1Agent,
        "action": RT1Action,
        "weight_keys": ["rt1main", "rt1simreal", "rt1multirobot", "rt1x"]
    },
    "octo": {
        "agent": OctoAgent,
        "action": OctoAction,
        "observation": VLA_SPACE,
        "variants": ["octo-small", "octo-base"]
    },
    "teleop": {
        "agent": TeleOpAgent,
        "action": RT1Action,
        "weight_keys": []
    }
}
