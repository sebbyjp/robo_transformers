from robo_transformers.abstract.agent import Agent
from robo_transformers.models.octo.action import OctoAction
from typing import Optional
import os
import cv2
import numpy as np
import numpy.typing as npt
from beartype import beartype
import math
from robo_transformers.replayer import Replayer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@beartype
class ReplayAgent(Agent):

  def __init__(
      self,
      record_dir: str = "episodes",
      weights_key: str = "episode_name",
  ) -> None:
    """Agent for octo model.

        Args:
            xyz_step (float, optional): Step size for xyz. Defaults to 0.02.
            rpy_step (float, optional): Step size for rpy. Defaults to PI/8.
        """
    self.replayer = iter(Replayer(weights_key, data_dir=record_dir))


  def act(
      self,
    *args,
    **kwargs,
  ) -> OctoAction:
    # Create observation of past `window_size` number of observations
    time_step = next(self.replayer)
    if time_step is None:
      raise StopIteration
    obs, act, reward, done = time_step
    return OctoAction(*[act['x'], act['y'], act['z'], act['roll'], act['pitch'], act['yaw'], act['grasp']])
