from robo_transformers.abstract.agent import Agent
from robo_transformers.models.octo.action import OctoAction
from typing import Optional
import os
import cv2
import numpy as np
import numpy.typing as npt
from beartype import beartype
import math
from robo_transformers.recorder import Replayer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@beartype
class ReplayAgent(Agent):

  def __init__(
      self,
      xyz_step: float = 0.01,
      rpy_step: float = math.pi / 8,
      record_dir: str = "episodes",
      weights_key: str = "episode_name",
  ) -> None:
    """Agent for octo model.

        Args:
            xyz_step (float, optional): Step size for xyz. Defaults to 0.02.
            rpy_step (float, optional): Step size for rpy. Defaults to PI/8.
        """
    self.xyz_step = xyz_step
    self.rpy_step = rpy_step
    self.recorder = None
    # self.recorder = Recorder(weights_key, data_dir=record_dir)
    self.replayer = iter(Replayer(weights_key, data_dir="episodes"))


  def act(
      self,
      instruction: str,
      image: npt.ArrayLike,
      image_wrist: Optional[npt.ArrayLike] = None,
  ) -> OctoAction:
    # Create observation of past `window_size` number of observations
    act = next(self.replayer)
    if act is not None:
      return OctoAction(*[act['x'], act['y'], act['z'], act['roll'], act['pitch'], act['yaw'], act['grasp']])
