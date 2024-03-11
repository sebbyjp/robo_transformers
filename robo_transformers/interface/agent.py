from beartype.typing import Any
from abc import ABC, abstractmethod
from beartype import beartype


# -----------------------------------------------------------------------------

@beartype
class Agent(ABC):
    '''Abstract class for an agent. Internal state such as history and past actions is kept in this class.
    '''

    @abstractmethod
    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def act(self, **kwargs) -> list:
        pass
