from dataclasses import dataclass, asdict
from beartype import beartype
from gym import spaces


@beartype
@dataclass
class Sample:
    '''Every sample is a dataclass that represents a sample from a gym Dict space.
    '''

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
        dict_with_nones = asdict(self)
        return {key: value for key, value in dict_with_nones.items() if value is not None}





