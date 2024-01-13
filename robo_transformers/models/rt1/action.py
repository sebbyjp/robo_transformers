from dataclasses import dataclass, field
from beartype import beartype
import numpy.typing as npt
import numpy as np
from robo_transformers.abstract.action import Action
import numpy.typing as npt
from typing import TypedDict


@beartype
class RT1ActionDictT(TypedDict):
    base_displacement_vector: npt.ArrayLike
    base_displacement_vertical: npt.ArrayLike
    gripper_closedness_action: npt.ArrayLike
    rotation_delta: npt.ArrayLike
    terminate_episode: npt.ArrayLike
    world_vector: npt.ArrayLike


@beartype
@dataclass
class RT1Action(Action):
    base_displacement_vector: np.ndarray = field(default_factory= lambda: np.array([0.0, 0.0]))
    base_displacement_vertical_rotation:  np.ndarray   = field(default_factory= lambda : np.array([0.0]))
    gripper_closedness_action: np.ndarray  = field(default_factory= lambda : np.array([1.0]))
    rotation_delta:  np.ndarray   = field(default_factory= lambda : np.array([0.0, 0.0, 0.0]))
    terminate_episode: np.ndarray   = field(default_factory= lambda : np.array([0,0,0]))
    world_vector:  np.ndarray  = field(default_factory= lambda : np.array([0.0, 0.0, 0.02]))