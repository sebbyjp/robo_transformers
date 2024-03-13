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
            d = OrderedDict(asdict(self))
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
        '''Return the action as an ordered dictionary.
        Returns:
            dict: The action as an ordered dictionary.
        '''
        dic = {key: value for key,value in OrderedDict(asdict(self)).items() if value is not None}
        for key, value in dic.items():
            if isinstance(value, Sample):
                dic[key] = value.todict()
        return dic





