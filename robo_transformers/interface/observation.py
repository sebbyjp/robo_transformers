from robo_transformers.interface import Sample
from dataclasses import dataclass
from gym import spaces
from beartype import beartype
from beartype.typing import SupportsFloat, Optional, SupportsInt
from enum import IntEnum

class Supervision(IntEnum):
    """Defines supported supervision types."""
    UNSUPERVISED = 0
    BINARY = 1 # e.g. "correct" or "incorrect"
    REWARD = 2 # continuous reward


@beartype
@dataclass
class Observation(Sample):
    '''Sample for an observation. Includes possible supervision and whether the episode is stopped.
    '''
    supervision: Optional[SupportsFloat] = None
    supervision_type: Supervision = Supervision.UNSUPERVISED
    stopped: bool = False # True if the episode is stopped or we were told to stop.

    def __post_init__(self):
        '''Post initialization.
        '''
        assert self.supervision_type in Supervision, f"Supervision type {self.supervision_type} not supported."
        if self.supervision_type == Supervision.BINARY:
          assert isinstance(self.supervision, SupportsInt), "Binary supervision must be an integer."
        if self.supervision_type == Supervision.REWARD:
          assert isinstance(self.supervision, SupportsFloat), "Reward supervision must be a float."

    def space(self) -> spaces.Dict:
        '''Return the observation space.
        Returns:
            spaces.Dict: The observation space.
        '''
        space = {
            "stopped": spaces.Discrete(2),
            "supervision_type": spaces.Discrete(len(Supervision)),
        }
        if self.supervision_type == Supervision.BINARY:
          space.update({"supervision": spaces.Discrete(2)})
        elif self.supervision_type == Supervision.REWARD:
          space.update({"supervision": spaces.Box(low=-1, high=1, shape=(), dtype=float)})
        return spaces.Dict(space)
        