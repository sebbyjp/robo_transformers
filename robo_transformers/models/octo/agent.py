from robo_transformers.abstract.agent import Agent
from robo_transformers.models.octo.action import OctoAction
from robo_transformers.models.rt1.action import RT1Action
from typing import Optional
import jax
import os
import cv2
import numpy as np
import numpy.typing as npt
from octo.model.octo_model import OctoModel
from beartype import beartype
from absl import logging
from PIL import Image
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@beartype
class OctoAgent(Agent):
    def __init__(self, weights_key: str = 'octo-small', window_size: int = 2, num_additional_future_actions: int = 3) -> None:
        '''Agent for octo model.

        Args:
            weights_key (str, optional): octo-small or octo-base. Defaults to 'octo-small'.
            window_size (int, optional): Number of past observations to use for inference. Must be <= 2. Defaults to 2.
            num_additional_future_actions (int, optional): Number of additional future actions to return. Defaults to 0.
        '''
        print('Loading from {}'.format(weights_key))
        self.model: OctoModel = OctoModel.load_pretrained("hf://rail-berkeley/" + weights_key)
        self.image_history = [] # Chronological order.
        self.image_wrist_history = [] # Chronological order.    
        self.window_size = window_size
        self.output_buffer = []
        assert num_additional_future_actions in [0,1,2,3,4] # Max actions returned by model.
        self.num_predicted_actions = 1 + num_additional_future_actions


    
    def act(self, instruction: str, image: npt.ArrayLike, image_wrist: Optional[npt.ArrayLike] = None, mean_action: Optional[npt.ArrayLike] = None, std_action: Optional[npt.ArrayLike] = None) -> OctoAction:

        # Create observation of past `window_size` number of observations
        image = cv2.resize(np.array(image, dtype=np.uint8), (256, 256))
        self.image_history.append(image)
        if len(self.image_history) > self.window_size:
            self.image_history.pop(0)
        
        images = np.stack(self.image_history)[None]
        np.expand_dims(images, axis=0)
        observation = {"image_primary": images, "pad_mask": np.full((1, images.shape[1]), True, dtype=bool)}
        
        if image_wrist is not None:
            image_wrist = cv2.resize(np.array(image_wrist, dtype=np.uint8), (128, 128))
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
                Image.fromarray(self.image_wrist_history[i]).save('image_wrist{}.png'.format(i))

        # Run inference
        if len(self.output_buffer) == 0:
            task = self.model.create_tasks(texts=[instruction])
            norm_actions = self.model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
            norm_actions = norm_actions[0]   # remove batch

            # Unormalize using bridge dataset if not specified
            if mean_action is None:
                mean_action = self.model.dataset_statistics["fractal20220817_data"]['action']['mean']
                # mean_action = np.array([0.05, -0.01, 0.01, 0.0, -0.05, 0.0, 0.0])
                # mean_action = np.array([0.01, -0.002, 0.0, 0.0, -0.02, 0.0, 0.0])
            if std_action is None:
                std_action =  self.model.dataset_statistics["fractal20220817_data"]['action']['std']
                # std_action = np.array([0.2, 0.2, 0.1, 0.2, 0.2, 0.2, 1.5])
                # std_action = np.array([0.045,0.031, 0.015, 0.02, 0.02, 0.02, 10.0])
            actions = norm_actions * std_action + mean_action

            for a in actions[:self.num_predicted_actions]:
                self.output_buffer.append(np.array(a).squeeze())
        else:
            print('Using buffer')
        action = self.output_buffer.pop(0)

        # Convert absolute grasp to relative grasp
        action[-1] = action[-1] * 2 - 1

        return OctoAction(x=float(action[0]), y=float(action[1]), z=float(action[2]), roll=float(action[3]), pitch=float(action[4]), yaw=float(action[5]), grasp=float(action[6]))
        # rt1_action = RT1Action(world_vector = action[0:3], rotation_delta=action[3:6], gripper_closedness_action=np.array(action[6]))
        #   action = np.sum(np.array(actions), axis = 0).squeeze()

        # return rt1_action

    