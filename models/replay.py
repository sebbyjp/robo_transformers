from robo_transformers.interface.agent import Agent
from beartype.typing import Any, Optional
from gym import spaces
from beartype import beartype
from robo_transformers.data.data_util import Replayer


@beartype
class ReplayAgent(Agent):

  def __init__(
      self,
      from_path: str,
      action_space: spaces.Space,
      observation_space: spaces.Space,
      **kwargs
  ) -> None:
    """Agent for octo model.

        Args:
            xyz_step (float, optional): Step size for xyz. Defaults to 0.02.
            rpy_step (float, optional): Step size for rpy. Defaults to PI/8.
        """
    self.replayer = iter(Replayer(from_path, observation_space, action_space, **kwargs))
    self.observation_space = observation_space
    self.action_space = action_space


  def act(
      self,
     **kwargs
  ) -> list:
    return [next(self.replayer)]