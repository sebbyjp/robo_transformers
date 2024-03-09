from enum import IntEnum
from robo_transformers.interface import Sample
from gym import spaces
from beartype import beartype
from dataclasses import dataclass

class Control(IntEnum):
    """Defines supported action control types in the world frame."""
    UNSPECIFIED = 0
    ABSOLUTE = 1 # Absolute control in the world frame.
    RELATIVE = 2 # Relative control to its previous value in the world frame.
    VELOCITY = 3
    EFFORT = 4

@beartype
@dataclass
class ControlAction(Sample):
    '''Action sample from a gym Dict space representing a control signal.
    '''
    control_type: Control = Control.UNSPECIFIED
    def space(self) -> spaces.Dict:
        '''Return the action space.
        Returns:
            spaces.Dict: The action space.
        '''
        return spaces.Dict({
            'control_type': spaces.Discrete(len(Control)),
        })
