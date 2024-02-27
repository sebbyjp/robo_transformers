from robo_transformers.abstract.agent import Agent
from robo_transformers.models.octo.action import OctoAction
from typing import Optional
import os
import cv2
import numpy as np
import numpy.typing as npt
from beartype import beartype
import math
from robo_transformers.recorder import Recorder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@beartype
class TeleOpAgent(Agent):

  def __init__(
      self,
      xyz_step: float = 0.01,
      rpy_step: float = math.pi / 8,
      record_dir: str = "episodes",
      weights_key: str = "episode_name",
  ) -> None:
    """Agent for octo model.

        Args:
            xyz_step (float, optional): Step size for xyz. Defaults to 0.01.
            rpy_step (float, optional): Step size for rpy. Defaults to PI/8.
        """
    self.xyz_step = xyz_step
    self.rpy_step = rpy_step
    self.recorder = Recorder(weights_key, data_dir=record_dir)
    self.buffer = []
    self.erased_last = False
    self.last_erased = None
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
            E = erase last time step \n
            - = decrease step size by factor of 5\n
            + = increase step size by factor of 5\n

            Type each command as many times as needed. Each will correspond to a single step (default xyz: 0.01, default rpy: PI/8.\n
            Press enter to continue.\n
            """)
    should_save = True
    grasp = 0.
    if "q" in value:
      grasp = -1.
    elif "e" in value:
      grasp = 1.
    else:
      grasp = 0.
    if "E" in value and len(self.buffer) > 0:
      confirmation = input("""
        WARNING: Detected E which means erase last time step. This action will be ignored. Press c to confirm or enter to cancel.\n
        """
      )
      if "c" in confirmation:
        confirmation = input("""
          WARNING:  Should I send the reverse
        of the erased action before recording continues to resume the same state? Y/n.\n
          """
        )
        self.last_erased = self.buffer.pop()
        if 'y' in confirmation.lower():
          print("""
            ***Note: Last time step was erased. Sending inverse of erased action.
          """)
          self.erased_last = True
          return -self.last_erased[-3], 0, 0, False

    reward = float("c" in value)
    done = float(reward or "f" in value)

    xyz_step = self.xyz_step
    rpy_step = self.rpy_step
    if "-" in value:
      xyz_step /= 5.
      rpy_step /= 5.
    elif "+" in value:
      xyz_step *= 5.
      rpy_step *= 5.

    xyz = xyz_step * np.array([
        value.count("w") - value.count("s"),
        value.count("a") - value.count("d"),
        value.count("x") - value.count("z"),
    ])
    rpy = rpy_step * np.array([
        value.count("D") - value.count("A"),
        value.count("S") - value.count("W"),
        value.count("Z") - value.count("X"),
    ])
    action = np.concatenate([xyz, rpy, [grasp]])
    return action, reward, done, should_save

  def act(
      self,
      instruction: str,
      image: npt.ArrayLike,
      image_wrist: Optional[npt.ArrayLike] = None,
  ) -> OctoAction:
    image = cv2.resize(np.array(image, dtype=np.uint8), (224, 224))
    if image_wrist is not None:
      image_wrist = cv2.resize(np.array(image_wrist, dtype=np.uint8), (128, 128))
    action, reward, done, should_save = self.process_input()

    print("action: ", action)
    if len(self.buffer) > 0:
      image, image_wrist, instruction, last_action, reward, done = self.buffer.pop()
      print("recording action: ", last_action)

      # Convert absolute grasp to relative grasp.
      grasp = (last_action[6] + 1) / 2. if last_action[6] != 0 else self.last_grasp
      print('recording grasp: ', grasp)
      self.last_grasp = grasp

      self.recorder.record(
          observation={
              "image_primary": image,
              "image_secondary": image_wrist,
              "language_instruction": instruction.encode(),
          },
          action={
              "left_hand": {
                  "x": last_action[0],
                  "y": last_action[1],
                  "z": last_action[2],
                  "roll": last_action[3],
                  "pitch": last_action[4],
                  "yaw": last_action[5],
                  "grasp": grasp,
                  "control_type": 0
              }
          },
          reward=reward,
          done=done,
      )
      if done:
        self.recorder.close(save_frames=True, save_stats=True)
    if should_save:
      self.buffer.append((image, image_wrist, instruction, action, reward, done))
    return OctoAction(*action)
