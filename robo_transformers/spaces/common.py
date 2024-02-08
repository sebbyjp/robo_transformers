from gym.spaces import Dict, Box, Text, Discrete
from enum import IntEnum
import numpy as np


class ActionControl(IntEnum):
    """Defines supported action control types."""
    POSE_DELTA = 0
    VELOCITY = 1
    EFFORT = 2


EEF_ACTION_SPACE = Dict({
        'x': Box(low=-1, high=1, shape=(), dtype=float),
        'y': Box(low=-1, high=1, shape=(), dtype=float),
        'z': Box(low=-1, high=1, shape=(), dtype=float),
        'roll': Box(low=-1, high=1, shape=(), dtype=float),
        'pitch': Box(low=-1, high=1, shape=(), dtype=float),
        'yaw': Box(low=-1, high=1, shape=(), dtype=float),
        'grasp': Box(low=-1, high=1, shape=(), dtype=float),
        'control_type': Discrete(len(ActionControl)), # POSE_DELTA, VELOCITY, EFFORT
    })



BASIC_BIMANUAL_ACTION_SPACE = Dict({
    'left_hand': EEF_ACTION_SPACE,
    'right_hand': EEF_ACTION_SPACE,
    'base_displacement':  Dict({
        'x': Box(low=-1, high=1, shape=(), dtype=float),
        'y': Box(low=-1, high=1, shape=(), dtype=float),
        'control_type': Discrete(len(ActionControl)), # POSE_DELTA, VELOCITY, EFFORT
    }),
    'vertical_rotation': Dict({
        'yaw': Box(low=-3.14, high=3.14, shape=(), dtype=float),
        'control_type': Discrete(len(ActionControl)), # POSE_DELTA, VELOCITY, EFFORT
    })
})


BASIC_VISION_LANGUAGE_OBSERVATION_SPACE =  Dict({
    'image_primary': Box(low=0, high=255, shape=(224,224,3), dtype=np.uint8),
    'image_secondary': Box(low=0, high=255, shape=(128,128,3), dtype=np.uint8),
    'language_instruction': Text(100),
})
