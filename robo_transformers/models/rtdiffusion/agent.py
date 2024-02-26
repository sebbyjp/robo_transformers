from absl import logging
from robo_transformers.abstract.agent import Agent
from robo_transformers.models.rt1.action import RT1Action
from typing import Optional
import os
import cv2
import numpy as np
import numpy.typing as npt
from beartype import beartype
from robo_transformers.oxe_torch.rtx1 import RTX1, FilmViTConfig, RT1Config
from robo_transformers.oxe_torch.action_tokenization import RTX1ActionTokenizer
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from einops import reduce, rearrange
from PIL import Image

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@beartype
class RTDiffusion(Agent):

  def __init__(self,
               saved_model_path:
               str = '/mbodi_ws/src/robo_transformers/checkpoints/mt1/pretrained_step92k.ckpt',
               num_additional_future_actions=4,
               reduction='mean',
               *args,
               **kwargs) -> None:
    '''Agent for octo model.
		Args:
			weights_key (str, optional): octo-small or octo-base. Defaults to 'octo-small'.
			window_size (int, optional): Number of past observations to use for inference. Must be <= 2. Defaults to 2.
			num_additional_future_actions (int, optional): Number of future actions to return. Defaults to 1.
		'''
    if 'weights_key' in kwargs:
      saved_model_path = kwargs['weights_key']

    self.model: RTX1 = RTX1(rt1_config=RT1Config(use_attn_conditioner=False,
                                                 causal_attention=False),
                            vit_config=FilmViTConfig(pretrained=True))

    if saved_model_path:
      print('Loading from {}'.format(saved_model_path))
      if (saved_model_path.endswith('.ckpt')):
        state_dict = torch.load(saved_model_path, map_location='cpu')['state_dict']
        print('Loaded state dict', state_dict.keys())
        consume_prefix_in_state_dict_if_present(state_dict, 'model.')
      else:
        state_dict = torch.load(saved_model_path, map_location='cpu')
  
      self.model.load_state_dict(state_dict)
      print('Loaded model')

    self.image_history = []  # Chronological order.
    self.tokenizer = RTX1ActionTokenizer()
    self.step_num = 0

    self.output_buffer = []
    assert num_additional_future_actions in [0, 1, 2, 3, 4, 5]  # Max actions returned by model.
    self.num_predicted_actions = 1 + num_additional_future_actions

    self.reduction = reduction

  def act(self, instruction: str, image: npt.ArrayLike) -> RT1Action:
    print(instruction)
    # Create observation of past `window_size` number of observations
    image = cv2.resize(np.array(image, dtype=np.uint8), (224, 224))

    image = rearrange(torch.as_tensor(np.array(image) / 255., dtype=torch.float), 'h w c -> c h w')
    if len(self.image_history) == 0:
      for i in range(6):
        self.image_history.append(torch.clone(image))
        # self.image_history.append(torch.zeros_like(image))

    self.image_history.append(image)
    if len(self.image_history) > 6:
      self.image_history.pop(0)
    # Stack images and add batch dimension.
    images = torch.stack(self.image_history, axis=0).unsqueeze(0)

    if logging.level_debug():
      for i in range(6):
        Image.fromarray(np.array(images[0, i, :, :, :].permute(1, 2, 0) * 255,
                                 dtype=np.uint8)).save('rt1_image_{}.png'.format(i))

    if len(self.output_buffer) == 0:
      with torch.no_grad():
        actions_out = self.model.run(images, [instruction], cond_scale=1.0)
        actions_out = rearrange(actions_out,
                                '1 f a bins -> f a bins')[:self.num_predicted_actions, :, :]
        if self.reduction == 'mean' and self.num_predicted_actions == 1:
          actions_out = reduce(actions_out, 'f a bins -> 1 a bins', 'mean')
        for i in range(self.num_predicted_actions):
          action_tokens = torch.argmax(actions_out[i, :, :], -1)
          # Undo integer truncation
          action_tokens = action_tokens + 1 * (action_tokens > 0)
          action = np.round(self.tokenizer.detokenize_vec(action_tokens).numpy(), 2)
          action[6] = action[6] * 2 - 1
          self.output_buffer.append(action)
    else:
      print('Using buffer')

    action = self.output_buffer.pop(0)
    print(action)
    return RT1Action(world_vector=action[0:3],
                     rotation_delta=action[3:6],
                     gripper_closedness_action=np.array(action[6]))
