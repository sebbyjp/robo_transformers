from robo_transformers.abstract.agent import Agent
from robo_transformers.models.octo.action import OctoAction
from typing import Optional
import os
import cv2
import numpy as np
import numpy.typing as npt
from beartype import beartype
from absl import logging
from PIL import Image
import math
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

PICK_COKE_CAN= [
     [0.100000, -0.040000, 0.040000, 0.000000, 0, 0.000000, 1.0],

]


@beartype
class TeleOpAgent(Agent):

    def __init__(self,
                 weights_key: str = '',
                 window_size: int = 2,
                 xyz_step: float = 0.01,
                 rpy_step: float = math.pi / 8) -> None:
        '''Agent for octo model.

        Args:
            weights_key (str, optional): octo-small or octo-base. Defaults to 'octo-small'.
            window_size (int, optional): Number of past observations to use for inference. Must be <= 2. Defaults to 2.
            num_future_actions (int, optional): Number of future actions to return. Defaults to 1.
            xyz_step (float, optional): Step size for xyz. Defaults to 0.02.
            rpy_step (float, optional): Step size for rpy. Defaults to PI/8.
        '''
        self.image_history = []  # Chronological order.
        self.image_wrist_history = []  # Chronological order.
        self.window_size = window_size
        self.output_buffer = []
        self.xyz_step = xyz_step
        self.rpy_step = rpy_step

    def act(self,
            instruction: str,
            image: npt.ArrayLike,
            image_wrist: Optional[npt.ArrayLike] = None
            ) -> OctoAction:

        # Create observation of past `window_size` number of observations
        image = cv2.resize(np.array(image, dtype=np.uint8), (256, 256))
        self.image_history.append(image)
        if len(self.image_history) > self.window_size:
            self.image_history.pop(0)

        images = np.stack(self.image_history)[None]
        np.expand_dims(images, axis=0)
        observation = {
            "image_primary": images,
            "pad_mask": np.full((1, images.shape[1]), True, dtype=bool)
        }

        if image_wrist is not None:
            image_wrist = cv2.resize(np.array(image_wrist, dtype=np.uint8),
                                     (128, 128))
            self.image_wrist_history.append(image_wrist)
            if len(self.image_wrist_history) > self.window_size:
                self.image_wrist_history.pop(0)

            # Add wrist image to observation
            image_wrists = np.stack(self.image_wrist_history)[None]
            np.expand_dims(image_wrists, axis=0)
            observation["image_wrist"] = image_wrists

        if logging.level_debug():
            for i, image in enumerate(self.image_history):
                Image.fromarray(image).save('image{}.png'.format(i))
                Image.fromarray(self.image_wrist_history[i]).save(
                    'image_wrist{}.png'.format(i))

        value = input(
            '''
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

            Type each command as many times as needed. Each will correspond to a single step (default xyz: 0.02, default rpy: PI/8.\n
            Press enter to continue.\n
            '''
        )

        grasp = 0
        if 'q' in value:
            grasp = -1
        elif 'e' in value:
            grasp = 1
        else:
            grasp = 0
        xyz = self.xyz_step * np.array([value.count('w') - value.count('s'),
                        value.count('a') - value.count('d'),
                        value.count('x') - value.count('z')])
        rpy = self.rpy_step * np.array([value.count('D') - value.count('A'),
                        value.count('S') - value.count('W'),
                        value.count('Z') - value.count('X')])
        action = np.concatenate([xyz, rpy, [grasp]])


        return OctoAction(*action)
