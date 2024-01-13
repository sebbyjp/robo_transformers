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
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@beartype
class OctoAgent(Agent):
    def __init__(self, weights_key: str = 'octo-small', window_size: int = 2) -> None:
        self.model: OctoModel = OctoModel.load_pretrained("hf://rail-berkeley/" + weights_key)
        self.image_history = [] # Chronological order.
        self.image_wrist_history = [] # Chronological order.    
        self.window_size = window_size

    
    def act(self, instruction: str, image: npt.ArrayLike, image_wrist: Optional[npt.ArrayLike] = None) -> RT1Action:
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
            
            image_wrists = np.stack(self.image_wrist_history)[None]
            np.expand_dims(image_wrists, axis=0)
            observation["image_wrist"] = image_wrists

        task = self.model.create_tasks(texts=[instruction])
      # this returns *normalized* actions --> we need to unnormalize using the dataset statistics
        norm_actions = self.model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
        norm_actions = norm_actions[0]   # remove batch
        actions = (
            norm_actions *self. model.dataset_statistics["bridge_dataset"]['action']['std']
            + self.model.dataset_statistics["bridge_dataset"]['action']['mean']
        )
        action = np.array(actions[0]).squeeze()
        #   action = np.sum(np.array(actions), axis = 0).squeeze()
        rt1_action = RT1Action(world_vector=action[0:3], rotation_delta=np.array([action[5], action[4], action[3]]), gripper_closedness_action=np.array(action[6]))

        return rt1_action

    