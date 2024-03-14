from einops import repeat
import torch

from torch.utils.data import IterableDataset

import numpy as np

import h5py
from torchvision import transforms

class MbodiedDataLoader(IterableDataset):

  def __init__(self,
               hdf5_file,
               data_augmentation=True,
               random_erasing=False,
               window_size=6,
               shuffle=False,
               seed=42,
               future_action_window_size=0,
               world_size=1,
               rank=0):
    """
        Args:
            hdf5_file (string): Path to the hdf5 file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
    self.world_size = world_size
    self.rank = rank

    self.file = h5py.File(hdf5_file, 'r')
    self.transform = lambda x: x
    if data_augmentation:
      # Define your transformations / augmentations
      transform = [
          transforms.ToTensor(),
          transforms.RandomApply([
              transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
              transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
              # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
          ])
      ]
      if random_erasing:
          transform.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.2, 2.0), value=0, inplace=False))
      self.transform = transforms.Compose(transform)

    self.window_size = window_size
    self.future_action_window_size = future_action_window_size
    self.shuffle = shuffle
    self.seed = seed

    self.length = self.file.attrs['size']
    self.idxs = np.arange(0, self.length)

    # if self.embed_instructions:
    #   self.text_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    if self.shuffle:
      np.random.seed(self.seed)
      np.random.shuffle(self.idxs)

  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    if self.shuffle:
      np.random.seed(self.seed)
      np.random.shuffle(self.idxs)

    mod = self.world_size
    shift = self.rank

    if worker_info:
      mod *= worker_info.num_workers
      shift = self.rank * worker_info.num_workers + worker_info.id

    for idx in self.idxs:
      if self.shuffle:
        np.random.seed(self.seed)
        np.random.shuffle(self.idxs)
      if (idx + shift) % mod == 0:
        observation = torch.zeros((self.window_size, 3, 224, 224), dtype=torch.float32)
        action = torch.zeros((self.window_size + self.future_action_window_size, 7), dtype=torch.float32)

        if idx < self.window_size - 1:
          observation[:self.window_size - idx - 1, :, :, :] = repeat(self.transform(
              self.file['observation/image'][idx]),
                                                                     'c h w -> f c h w',
                                                                     f=self.window_size - idx - 1)
          observation[self.window_size - idx - 1:, :, :, :] = torch.stack(
              [self.transform(self.file['observation/image'][i]) for i in range(0, idx + 1)])

          action[self.window_size - idx - 1:self.window_size, :] = torch.tensor([[
              *self.file['action/left_hand/pose/xyz'][i], 
              *self.file['action/left_hand/pose/rpy'][i],
              self.file['action/left_hand/grasp/value'][i],
              self.file['action/left_gripper/grasp/value'][i],
          ] for i in range(0, idx + 1)])
        else:
          observation = torch.stack([
              self.transform(self.file['observation/image'][i])
              for i in range(idx - self.window_size + 1, idx + 1)
          ])
          action[:self.window_size, :] = torch.tensor([[
              *self.file['action/left_hand/pose/xyz'][i], 
              *self.file['action/left_hand/pose/rpy'][i],
              self.file['action/left_hand/grasp/value'][i],
          ] for i in range(idx - self.window_size + 1, idx + 1)])
        if idx + 1 > self.length - self.future_action_window_size and idx + 1 < self.length:
          action[self.window_size:self.window_size + self.length - idx - 1, :] = torch.tensor([[
              *self.file['action/left_hand/pose/xyz'][i], 
              *self.file['action/left_hand/pose/rpy'][i],
              self.file['action/left_hand/grasp/value'][i],
              self.file['action/left_hand/grasp/value'][i],
          ] for i in range(idx + 1, self.length)])
        else:
          action[self.window_size:, :] = torch.tensor([[
              *self.file['action/left_hand/pose/xyz'][i], 
              *self.file['action/left_hand/pose/rpy'][i],
              self.file['action/left_hand/grasp/value'][i],
              self.file['action/left_hand/grasp/value'][i],
          ] for i in range(idx + 1, idx + 1 + self.future_action_window_size)])

        yield {
            'observation': {
                "image_primary": observation
            },
            'action': action,
            'language_instruction': self.file['observation/instruction'][idx][0].decode(),
        }
