from robo_transformers.interface import Agent, Control, Supervision
from robo_transformers.data_util import Recorder
from robo_transformers.common.observations import ImageInstruction
from robo_transformers.common.actions import GripperBaseControl, GripperControl, JointControl, PoseControl
from beartype.typing import Any, Optional
import os
import cv2
import numpy as np
import numpy.typing as npt
from beartype import beartype
import math
from gym import spaces

from agents_corp.rt_genie.smart_vlacm import SVLACM
from agents_corp.rt_genie.motion_control import MotionController, LocobotJoyController, LocobotControl

os.environ["TOKENIZERS_PARALLELISM"] = "false"



@beartype
class LanguageAgent(Agent):

  def __init__(
      self,
      name: str,
      data_dir: str = "episodes",
      xyz_step: float = 0.01,
      rpy_step: float = math.pi / 8,
      **kwargs
  ) -> None:
    """Agent for teleop psudo model.

        Args:
            xyz_step (float, optional): Step size for xyz. Defaults to 0.02.
            rpy_step (float, optional): Step size for rpy. Defaults to PI/8.
        """
    if 'observation_space' in kwargs:
      self.observation_space = kwargs['observation_space']
      del kwargs['observation_space']
    if 'action_space' in kwargs:
      self.action_space = kwargs['action_space']
      del kwargs['action_space']
    else:
      self.observation_space = ImageInstruction().space()
      self.action_space = LocobotControl().space()
    self.xyz_step = xyz_step
    self.rpy_step = rpy_step
    self.recorder = Recorder(name,out_dir=data_dir, observation_space=self.observation_space, action_space=self.action_space, **kwargs)
    self.last_grasp = 0
    self.motion_controller = LocobotJoyController()
    self.vlacm = SVLACM()

  def act(
      self,
      instruction: str,
      image: npt.ArrayLike,
  ) -> list:
    # Create observation of past `window_size` number of observations
    image = cv2.resize(np.array(image, dtype=np.uint8),(640, 480))
    self.vlacm.last_image_seen = image
    motion_controls, language_controls, last_seen_image, instruction =  self.vlacm.act()

    gripper_base_controls = self.motion_controller.to_gripper_base_controls(motion_controls)

    for control, language_control in zip(gripper_base_controls, language_controls):
        control: LocobotControl = control
        print('control from la %s is %s ',language_control,  control)
        # Convert absolute grasp to relative grasp.
        grasp = control.left_gripper.grasp.value
        if grasp == 0:
            grasp = self.last_grasp
        else:
            grasp = (grasp + 1) / 2
        print('recording grasp: ', grasp)
        
        self.last_grasp = grasp
        self.recorder.record(
            observation= ImageInstruction(instruction=instruction, image=last_seen_image).todict(),
            action=control.todict(),
        )
   
    return [control.todict() for control in gripper_base_controls]
