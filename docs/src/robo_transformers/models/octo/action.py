from dataclasses import dataclass
from beartype import beartype
from robo_transformers.abstract.action import Action
from typing import TypedDict



@beartype
class OctoActionDictT(TypedDict):
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float
    grasp: float



@beartype
@dataclass
class OctoAction(Action):
    '''End effector pose deltas.
    '''
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    grasp: float = 0.0  # How much to open or close the gripper.

    @classmethod
    def from_jax_array(cls, jax_array):
        return cls(*[float(x) for x in jax_array])



