from robo_transformers.abstract.agent import Agent
from robo_transformers.models.rt1.action import RT1Action
from typing import Optional
import os
import cv2
import numpy as np
import numpy.typing as npt
from beartype import beartype
from rtx.rtx1 import RTX1, FilmViTConfig, RT1Config
from rtx.action_tokenization import RTX1ActionTokenizer
import torch
from PIL import Image
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@beartype
class RTDiffusion(Agent):

    def __init__(self,
                 saved_model_path: str = '') -> None:
        '''Agent for octo model.

        Args:
            weights_key (str, optional): octo-small or octo-base. Defaults to 'octo-small'.
            window_size (int, optional): Number of past observations to use for inference. Must be <= 2. Defaults to 2.
            num_future_actions (int, optional): Number of future actions to return. Defaults to 1.
        '''
        self.model = RTX1( rt1_config = RT1Config(use_attn_conditioner=True),vit_config=FilmViTConfig(pretrained=False))
        if saved_model_path:
            print('Loading from {}'.format(saved_model_path))
            state_dict = torch.load(saved_model_path, map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.image_history = []  # Chronological order.
        self.tokenizer = RTX1ActionTokenizer()
        with torch.no_grad():
            video = torch.zeros(6, 3, 224, 224)
        for i in range(6):
            self.image_history.append(video[i,:,:,:])


    def act(self,
            instruction: str,
            image: npt.ArrayLike) -> RT1Action:
        # Create observation of past `window_size` number of observations
        image = cv2.resize(np.array(image, dtype=np.uint8), (224, 224))
        # h,w,c --> c,h,w
        image = torch.as_tensor(np.array(image) / 255., dtype=torch.float).permute(2, 0, 1)
        self.image_history.append(image)
        if len(self.image_history) > 6:
            self.image_history.pop(0)

        # Stack images and add batch dimension.
        images = torch.stack(self.image_history, axis=0).unsqueeze(0)
        # print('before save')
        # Image.fromarray(np.array(images[0,-1,:,:,:].permute(1,2,0)*255, dtype=np.uint8)).save('rt1_image.png')
        print('before run')
        # Remove batch and select action pred for last frame
        with torch.no_grad():
            action_pred = self.model.run(images, [instruction])[0,-1,:,: ]
        print('before get')
        _, action_tokens =torch.max(action_pred, -1)
        print('before detokenize')
        action = self.tokenizer.detokenize_vec(action_tokens).numpy()
        print(action)  
        return  RT1Action(world_vector = action[0:3], rotation_delta=action[3:6], gripper_closedness_action=np.array(action[6]))
