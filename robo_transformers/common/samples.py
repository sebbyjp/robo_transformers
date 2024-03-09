from robo_transformers.interface.sample import Sample
from typing import Union, Sequence, SupportsFloat
from gym import spaces
from dataclasses import dataclass, field
from beartype import beartype
import numpy as np

@beartype
@dataclass
class Discrete(Sample):
    '''Single discrete sample space.
    '''
    value: int = 0
    size: int = 2

    def space(self) -> spaces.Dict:
        return spaces.Dict({'value': spaces.Discrete(self.size)})

@beartype
@dataclass
class Continuous(Sample):
    '''Single continuous sample space.
    '''
    value: SupportsFloat = 0
    bounds: Union[Sequence[SupportsFloat], np.ndarray] = field(default_factory=lambda: [-1, 1])

    def space(self) -> spaces.Dict:
        return spaces.Dict({'value': spaces.Box(low=self.bounds[0], high=self.bounds[1], shape=(), dtype=float)})



@beartype
@dataclass
class Pose(Sample):
    '''Action for a 6D space representing x, y, z, roll, pitch, and yaw.
    '''
    xyz: Union[Sequence[SupportsFloat], np.ndarray] = field(default_factory=lambda: [0, 0, 0])
    rpy: Union[Sequence[SupportsFloat], np.ndarray] = field(default_factory=lambda: [0, 0, 0])
    xyz_bounds: Union[Sequence[SupportsFloat], np.ndarray] = field(default_factory=lambda: [-1, 1])
    rpy_bounds: Union[Sequence[SupportsFloat], np.ndarray] = field(default_factory=lambda: [-3.14, 3.14])

    def space(self):
        return spaces.Dict({
            'xyz': spaces.Box(low=self.xyz_bounds[0], high=self.xyz_bounds[1], shape=(3,), dtype=float),
            'rpy': spaces.Box(low=self.rpy_bounds[0], high=self.rpy_bounds[1], shape=(3,), dtype=float),
        })
        
@beartype
@dataclass
class PlanarDirection(Sample):
    '''Action for a 2D+1 space representing x, y, and yaw.
    '''
    xy: Union[Sequence[SupportsFloat], np.ndarray] = field(default_factory=lambda: [0, 0])
    yaw: Union[SupportsFloat, np.ndarray] = field(default_factory=lambda: [0])
    xy_bounds: Union[Sequence[SupportsFloat], np.ndarray] = field(default_factory=lambda: [-1, 1])
    yaw_bounds: Union[Sequence[SupportsFloat], np.ndarray] = field(default_factory=lambda: [-3.14, 3.14])

    def space(self):
        return spaces.Dict({
            'xy': spaces.Box(low=self.xy_bounds[0], high=self.xy_bounds[1], shape=(2,), dtype=float),
            'yaw': spaces.Box(low=self.yaw_bounds[0], high=self.yaw_bounds[1], shape=(), dtype=float),
        })
