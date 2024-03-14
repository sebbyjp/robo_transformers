from robo_transformers.interface import Agent, Control, Supervision
from robo_transformers.data.data_util import Recorder
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@beartype
class TeleOpAgent(Agent):

  def __init__(self,
               name: str,
               data_dir: str = "episodes",
               xyz_step: float = 0.01,
               rpy_step: float = math.pi / 8,
               observation_space: Optional[spaces.Space] = None,
                action_space: Optional[spaces.Space] = None,
               **kwargs) -> None:
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
      self.observation_space = ImageInstruction(supervision_type=Supervision.BINARY,
                                                supervision=1).space()
      self.action_space = GripperBaseControl().space()
    self.xyz_step = xyz_step
    self.rpy_step = rpy_step
    self.last_grasp = 0

  def process_input(self):
    value = input("""
            Please enter one or more values for the action. The following correspond to a pose delta in the world frame:\n
            w = forward x (+x) \n
            s = backward x (-x) \n
            a = left y (+y)\n
            d = right y (-y)\n
            x = up z (+z) \n
            z = down z (-z) \n
            q = close gripper\n
            e = open gripper\n
            shift+d = roll right (-r) \n
            shift+a = roll left (+r)\n
            shift+w = pitch up (-p)\n
            shift+s = pitch down (+p)\n
            shift+z = yaw left (+y)\n
            shift+x = yaw right (-y)\n
            c = complete episode successfully\n
            f = finish episode with failure\n
            u = undo last action\n

            Type each command as many times as needed. Each will correspond to a single step (default xyz: 0.01, default rpy: PI/8.\n
            Press enter to continue.\n
            """)

    grasp = 0.
    if "q" in value:
      grasp = -1.
    elif "e" in value:
      grasp = 1.
    reward = int("c" in value)
    done = bool(reward or "f" in value)

    xyz = self.xyz_step * np.array([
        value.count("w") - value.count("s"),
        value.count("a") - value.count("d"),
        value.count("x") - value.count("z"),
    ])
    rpy = self.rpy_step * np.array([
        value.count("D") - value.count("A"),
        value.count("S") - value.count("W"),
        value.count("Z") - value.count("X"),
    ])
    action = np.concatenate([xyz, rpy, [grasp]])
    return action, reward, done

  def act(
      self,
      instruction: str,
      image: npt.ArrayLike,
  ) -> list:
    # Create observation of past `window_size` number of observations
    image = cv2.resize(np.array(image, dtype=np.uint8), (640, 480))
    action, reward, done = self.process_input()

    print("action: ", action)
    grasp = action[6]

    # Convert absolute grasp to relative grasp.
    if grasp == 0:
      grasp = self.last_grasp
    else:
      grasp = (grasp + 1) / 2
    print('recording grasp: ', grasp)

    self.last_grasp = grasp
    action = GripperBaseControl(finish=done,
                                left_gripper=GripperControl(pose=PoseControl(xyz=action[:3],
                                                                             rpy=action[3:6]),
                                                            grasp=JointControl(
                                                                control_type=Control.ABSOLUTE,
                                                                value=grasp)))
    self.recorder.record(
        observation=ImageInstruction(instruction=instruction,
                                     image=image,
                                     supervision=int(reward),
                                     supervision_type=Supervision.BINARY).todict(),
        action=action.todict(),
    )

    return [action.todict()]
