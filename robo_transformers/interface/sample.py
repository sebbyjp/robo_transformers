from dataclasses import dataclass, asdict
from beartype import beartype
from gym import spaces
import numpy as np
from collections import OrderedDict

@beartype
@dataclass
class Sample:
    '''Every sample is a dataclass that represents a sample from a gym Dict space.
    '''

    @classmethod
    def fromdict(cls, d: dict):
        '''Create a sample from a dictionary.
        Args:
            d (dict): The dictionary.
        Returns:
            Sample: The sample.
        '''
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = cls.fromdict(value)
        return cls(**d)

    def flatten(self, d=None) -> list:
        '''Flatten the sample. 
        Returns:
            list: The flattened sample.
        '''
        if d is None:
            d = asdict(self)
        else:
            d = OrderedDict(d)

        def items():
            for key, value in d.items():
                if isinstance(value, dict):
                    yield from self.flatten(value)
                else:
                    yield value

        return np.array(items())

    def space(self) -> spaces.Dict:
        '''Return the action space.
        Returns:
            spaces.Dict: The action space.
        '''
        raise NotImplementedError

    def todict(self) -> dict:
        '''Return the action as a dictionary.
        Returns:
            dict: The action as a dictionary.
        '''
        dict_with_nones = OrderedDict(asdict(self))
        return {key: value for key, value in dict_with_nones.items() if value is not None}





