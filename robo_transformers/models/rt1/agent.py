from robo_transformers.abstract.agent import Agent
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.policies.tf_policy import TFPolicy
from robo_transformers.models.rt1.inference import load_rt1, inference, embed_text
from robo_transformers.models.rt1.action import RT1Action
from typing import Optional
import numpy.typing as npt
import cv2
import numpy as np
import torch
from beartype import beartype
from einops import repeat, rearrange
import sys
sys.path.append('/simply_ws/src/robo_transformers/robo_transformers')
from robo_transformers.oxe_torch.rt1.maruya24_rt1.tokenizers.utils import batched_space_sampler, np_to_tensor
from robo_transformers.oxe_torch.rt1.maruya24_rt1.transformer_network import TransformerNetwork
from robo_transformers.oxe_torch.rt1.maruya24_rt1.transformer_network_test_set_up import state_space_list
import copy
from robo_transformers.recorder import Replayer
from PIL import Image

from gym import spaces
from collections import OrderedDict


def dict_to_device(dict_obj, device):
    """
    put all the values in the [dict_obj] to [device]
    """
    for k, v in dict_obj.items():
        v = torch.as_tensor(v)
        assert isinstance(v, torch.Tensor)
        dict_obj[k] = v.to(device)
    return dict_obj



action_space = spaces.Dict(
        OrderedDict(
            [
                ("terminate_episode", spaces.Discrete(4)),
                (
                    "world_vector",
                    spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                ),
                (
                    "rotation_delta",
                    spaces.Box(
                        low=-np.pi , high=np.pi, shape=(3,), dtype=np.float32
                    ),
                ),
                (
                    "gripper_closedness_action",
                    spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
                ),
            ]
        )
    )
# Agent for RT1
@beartype
class RT1Agent(Agent):
    def __init__(self, weights_key: str, weights_path: str = '/simply_ws/src/RT-X/checkpoints/step450.pt', **kwargs) -> None:

        self.weights_key = weights_key

        self.policy_state: Optional[dict]   = None
  
        network_configs = {
                "vocab_size": 256,
                "token_embedding_size_per_image": 512,
                "language_embedding_size": 512,
                "num_layers": 8,
                "layer_size": 128,
                "num_heads": 8,
                "feed_forward_size": 512,
                "dropout_rate": 0.1,
                "crop_size": 236,
                "use_token_learner": True,
                "time_sequence_length": 6,
                "num_encoders": 2,
                "using_proprioception": False,
                "input_tensor_space": None,
                "output_tensor_space": None,
                "cam_view": ["front"],
            }

        # Modify network configuration based on specific settings
        network_configs["time_sequence_length"] = 6
        network_configs["num_encoders"] = len(network_configs["cam_view"])
        network_configs["token_embedding_size"] = network_configs[
            "token_embedding_size_per_image"
        ] * len(network_configs["cam_view"])
        del network_configs["token_embedding_size_per_image"]
        del network_configs["cam_view"]
        network_configs["input_tensor_space"] = state_space_list()[0]
        network_configs["output_tensor_space"] = action_space


        if weights_key == "rt1pose":
            self.model = TransformerNetwork(**network_configs)
            self.policy_state = np_to_tensor(
                batched_space_sampler(
                    self.model._state_space,
                    batch_size=1,
                )
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if weights_path:
                print('Loading weights from ' + weights_path)
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            self.image_history = []  
            for i in range(6):
                self.image_history.append(torch.zeros(size=(224, 224, 3), dtype=torch.float, device=self.device))

        else:
            self.model: PyPolicy | TFPolicy = load_rt1(model_key=weights_key)
        self.step_num: int = 0
   
        

    def act(self, instruction: str, image: npt.ArrayLike, reward: float = 0.0) -> RT1Action:
        if self.weights_key == 'rt1pose':
            image = cv2.resize(np.array(image, dtype=np.uint8), (224, 224)) 
            self.image_history.append(torch.tensor(image / 255.0 , dtype=torch.float32, device=self.device))
            if len(self.image_history) > 6:
                self.image_history.pop(0)
            images = torch.stack(self.image_history)[None]
            print('instruction: ', instruction)
        
            # for i, image in enumerate(self.image_history):
            #     Image.fromarray(np.array(255 * images[0,i], dtype=np.uint8)).save('img{}.png'.format(i))

            video = rearrange(images.to(self.device), 'b f h w c -> b f c h w')

            obs = {'image': video[:,-1,:,:,:], 'natural_language_embedding': np.array(embed_text(instruction, 1))}

        
            # self.model.set_actions(dict_to_device({
            #     'terminate_episode': repeat(torch.ones((video.shape[0]), dtype=torch.long), 'b -> b f', f=video.shape[1]),
            #     'world_vector':    repeat(torch.zeros(1,3), 'b n  -> b f n', f=video.shape[1]),
            #     'rotation_delta':  repeat(torch.zeros(1,3), 'b n  -> b f n', f=video.shape[1]),
            #     'gripper_closedness_action': repeat(torch.zeros(1,1), 'b n  -> b f n', f=video.shape[1])
            # }, self.device))

            action, network_state = self.model(
                dict_to_device(obs, self.device),
                dict_to_device(self.policy_state, self.device),
            )

            self.policy_state = network_state
            action['gripper_closedness_action'] =  action['gripper_closedness_action'] * 2 - 1

            if self.step_num == 0:
                action['gripper_closedness_action'][0] = 1.0
            
            if self.replayer is not None:
                action = next(self.replayer)
                print(action)
                return RT1Action(world_vector=np.array([action['x'], action['y'], action['z']]), rotation_delta=np.array([action['roll'], action['pitch'], action['yaw']]), gripper_closedness_action=np.array(action['grasp']))
            else:
                print(action)
                return RT1Action.from_numpy_dict(action)
        else:
            # if self.weights_key == 'rt1x':
            #     image = image / 255.0
            # if self.step_num == 0:
            #     for i in range(12):
            #         action, next_state, _ = inference(instruction, image, self.step_num, reward, self.model, self.policy_state)
            #         self.policy_state = next_state
            image = np.array(image, dtype=np.uint8)
            action, next_state, _ = inference(instruction, image, self.step_num, reward, self.model, self.policy_state)
            self.policy_state = next_state
            return RT1Action.from_numpy_dict(action)

        self.step_num += 1
        return RT1Action(world_vector=np.array([action['x'], action['y'], action['z']]), rotation_delta=np.array([action['roll'], action['pitch'], action['yaw']]), gripper_closedness_action=np.array(action['grasp']))
