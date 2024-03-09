from robo_transformers.common.actions import GripperControl, PoseControl, GraspControl
from robo_transformers.common.observations import MultiImageInstruction, Image
from robo_transformers.interface.action import Control
from robo_transformers.interface import Agent

from typing import Any, Optional
import jax
import os
import cv2
import numpy as np
import numpy.typing as npt
from octo.model.octo_model import OctoModel
from beartype import beartype
from absl import logging
from PIL import Image as PILImage

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# class RT1Agent(Agent):
#   '''Inspired by Google's RT-1
#       An agent that can interpret an instruction and temporal sequence of up to three images per timestep
#       to navigate its body in 2D (planar) space, rotate about the vertical axis, and control two hands in 7D space
#       which correspond to the x, y, z, roll, pitch, yaw, and openness of the hand.

#     '''

#   def __init__(self, model_uri: str,*, retention: int, image_height: int = 480, image_width: int = 640):
#     self.height = image_height
#     self.width = image_width
#     self.observation_space = ImageInstruction(instruction=' ', image=Image(height=self.height, width=self.width)).space()
#     self.action_space = MbodiedControl(actions=[
#         PlanarDirectionAction(Control.RELATIVE),
#         GripperControl(pose=PoseAction(Control.RELATIVE),
#                             grasp=GraspAction(Control.RELATIVE)),
#         Discrete(size=3)
#     ],
#                                        names=['base', 'gripper', 'done'],
#     ).space()
#     self.observation_buffer: list[ImageInstruction] = []
#     self.action_buffer = []
#     self.retention = retention
#     self.step_num = 0
#     self.policy_state = None

#     if model_uri == 'gs://rt1/rt1main':
#         self.policy: TFPolicy | PyPolicy = load_rt1(model_key='rt1x')

#     else:
#         raise ValueError('Model not found')

#   def act(self, *, image: np.Array = None, instruction: str = None) -> list[MbodiedControl]:
#     image = np.array(image, dtype=np.uint8)
#     self.observation_buffer.append(ImageInstruction(instruction=instruction, image=Image(pixels=image)))

#     if len(self.observation_buffer) > self.retention:
#         self.observation_buffer.pop(0)

#     action, next_state, _ = inference(instruction, image, self.step_num, 0, self.policy, self.policy_state)

#     self.policy_state = next_state
#     self.step_num += 1

#     return MbodiedControl(actions=[
#         PlanarDirectionAction(xy=np.array(action['base_displacement_vector']), yaw=np.array(action['base_displacement_vertical'])),
#         GripperControl(pose=PoseAction(xyz=np.array(action['world_vector']), rpy=np.array(action['rotation_delta'])),
#                             grasp=GraspAction(action['gripper_closedness_action'])),
#         Discrete(action['terminate_episode'])
#     ],
#                           names=['base', 'gripper', 'done']).todict()


@beartype
class OctoAgent(Agent):

  def __init__(self, model_uri: str, *, retention: int = 2, foresight: int = 3, **kwargs):
    '''Agent for octo model.

        Args:
            weights_key (str, optional): octo-small or octo-base. Defaults to 'octo-small'.
            window_size (int, optional): Number of past observations to use for inference. Must be <= 2. Defaults to 2.
            num_additional_future_actions (int, optional): Number of additional future actions to return. Defaults to 0.
        '''
    print('Loading from {}'.format("hf://rail-berkeley/octo-base"))
    self.policy: OctoModel = OctoModel.load_pretrained(model_uri)
    self.observation_space = MultiImageInstruction(
        instruction=' ', images=[Image(height=256, width=320),
                                 Image(height=128, width=128)]).space()
    self.action_space = GripperControl(pose=PoseControl(Control.RELATIVE),
                                       grasp=GraspControl(Control.ABSOLUTE, grasp_bounds=[0, 1.0])).space()
    self.observation_buffer: list[MultiImageInstruction] = []
    self.retention = retention
    self.foresight = foresight
    self.step_num = 0

  def act(self,
          *,
          instruction: str,
          image: npt.ArrayLike,
          image_wrist: Optional[npt.ArrayLike] = None,
          mean_action: Optional[npt.ArrayLike] = None,
          std_action: Optional[npt.ArrayLike] = None) -> list[Any]:

    # Create observation of past `window_size` number of observations
    image = cv2.resize(np.array(image, dtype=np.uint8), (256, 256))
    self.observation_buffer.append(
        MultiImageInstruction(instruction=instruction,
                              images=[Image(pixels=image),
                                      Image(pixels=image_wrist)]))
    if len(self.observation_buffer) > self.retention:
      self.observation_buffer.pop(0)

    images = np.stack([obs.images[0].pixels for obs in self.observation_buffer])[None]

    np.expand_dims(images, axis=0)
    observation = {
        "image_primary": images,
        "pad_mask": np.full((1, images.shape[1]), True, dtype=bool)
    }

    if image_wrist is not None:
      image_wrist = cv2.resize(np.array(image_wrist, dtype=np.uint8), (128, 128))
      self.observation_buffer[-1].images[1].pixels = image_wrist
      image_wrists = np.stack([obs.images[1].pixels for obs in self.observation_buffer])[None]
      np.expand_dims(image_wrists, axis=0)
      observation["image_wrist"] = image_wrists

    if logging.level_debug():
      for i, image in enumerate(self.image_history):
        Image.fromarray(image).save('image{}.png'.format(i))
        Image.fromarray(self.image_wrist_history[i]).save('image_wrist{}.png'.format(i))

    task = self.model.create_tasks(texts=[instruction])
    norm_actions = self.model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
    norm_actions = norm_actions[0]  # remove batch

    # Unormalize using bridge dataset if not specified
    if mean_action is None:
      mean_action = self.model.dataset_statistics["fractal20220817_data"]['action']['mean']
      # mean_action = np.array([0.05, -0.01, 0.01, 0.0, -0.05, 0.0, 0.0])
      # mean_action = np.array([0.01, -0.002, 0.0, 0.0, -0.02, 0.0, 0.0])
    if std_action is None:
      std_action = self.model.dataset_statistics["fractal20220817_data"]['action']['std']
      # std_action = np.array([0.2, 0.2, 0.1, 0.2, 0.2, 0.2, 1.5])
      # std_action = np.array([0.045,0.031, 0.015, 0.02, 0.02, 0.02, 10.0])
    actions = norm_actions * std_action + mean_action

    return [
        GripperControl(pose=PoseControl(Control.RELATIVE, xyz=actions[i, :3], rpy=actions[i, 3:6]),
                       grasp=GraspControl(Control.ABSOLUTE, actions[i, 6], grasp_bounds=[0,1.0])).todict() for i in range(self.foresight)
    ]
