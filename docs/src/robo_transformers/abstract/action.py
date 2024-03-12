from dataclasses import dataclass, asdict
from beartype import beartype
import numpy as np



@beartype
@dataclass
class Action:
    @classmethod
    def from_numpy_dict(cls, d: dict):
        return cls(**{k: np.squeeze(v.numpy(), axis=0) for k, v in d.items()})

    def make(self) -> dict:
        '''Return the action as a dictionary.
        Returns:
            dict: The action as a dictionary.
        '''
        return asdict(self)

