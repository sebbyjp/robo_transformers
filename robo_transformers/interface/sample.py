from dataclasses import dataclass, asdict
from beartype import beartype
from gym import spaces
import numpy as np
from collections import OrderedDict
from beartype.typing import SupportsFloat, Union, Sequence, Optional
from absl import logging
@beartype
@dataclass
class Sample:
    '''Every sample is a dataclass that represents a sample from a gym Dict space.
    '''
    
    @staticmethod
    def dict_of_lists_to_list_of_dicts(d: dict) -> list:
        '''Convert a dictionary of lists to a list of dictionaries.
        Args:
            d (dict): The dictionary.
        Returns:
            list: The list of dictionaries.
        '''
        keys = d.keys()
        return [dict(zip(keys, values)) for values in zip(*d.values())]
    
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
            elif isinstance(value, Union[Sequence, np.ndarray]):
                d[key] = np.array(value)
            else :
                d[key] = value
        return cls(**d)
    
    @classmethod
    def fromspace(cls, space: spaces.Dict):
        '''Create a sample from a space.
        Args:
            space (spaces.Dict): The space.
        Returns:
            Sample: The sample.
        '''
        d = {}
        for key, value in space.spaces.items():
            if isinstance(value, spaces.Dict):
                d[key] = cls.fromspace(value)
            elif isinstance(value, spaces.Box):
                d[key] = value.sample()
            elif isinstance(value, spaces.Discrete):
                d[key] = value.sample()
            else:
                raise ValueError(f'Unsupported space type: {type(value)}')
        return cls(**d)


    def flatten(self, without=(),d=None) -> Union[np.ndarray, Sequence]:
        '''Flatten the sample.  TODO: add support for removing fields or matching another sample
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

        # Create a one-dimensional list from the generator
        flat_list = [item for sublist in items() for item in sublist]
        return np.array(flat_list)
    
    def fromflat(self, flat: Union[Sequence, np.ndarray], d=None) -> None:
        '''Create a sample from a flattened sample.
        Args:
            flat (np.ndarray): The flattened sample.
        '''
        if d is None:
            d = OrderedDict(asdict(self))
        else:
            d = OrderedDict(d)

        def items():
            for key, value in d.items():
                if isinstance(value, dict):
                    yield key, self.fromflat(flat, value)
                else:
                    yield key, flat.pop(0)

        for key, value in items():
            setattr(self, key, value)


    def space(self) -> spaces.Dict:
        '''Return the action space.
        Returns:
            spaces.Dict: The action space.
        '''
        logging.warn('This is a default space based on the dataclass fields. Please override this method to define a custom space.')
        d = self.todict()
        space = {}
        for key, value in d.items():
            if isinstance(value, Sample):
                space[key] = value.space()
            elif isinstance(value, np.ndarray):
                space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype=value.dtype)
            elif isinstance(value, int):
                space[key] = spaces.Discrete(value)
            elif isinstance(value, float):
                space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            else:
                raise ValueError(f'Unsupported type: {type(value)}')

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





