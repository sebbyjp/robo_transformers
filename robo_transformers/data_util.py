from typing import Any
import h5py
from h5py import string_dtype
import numpy as np
import os
from gym import spaces
from datetime import datetime
from robo_transformers.interface import Sample
from PIL import Image
from pathlib import Path

def create_dataset_for_space_dict(space_dict: spaces.Dict, group: h5py.Group, num_steps: int = 10):
    for key, space in space_dict.items():
        print('key: ', key, ', value: ', space)
        if isinstance(space, spaces.Dict):
            subgroup = group.create_group(key)
            create_dataset_for_space_dict(space, subgroup, num_steps)
        else:
            shape = space.shape if space.shape else (num_steps,)
            dtype = space.dtype if space.dtype != str else string_dtype()
            group.create_dataset(key, (num_steps, *shape), dtype=dtype, maxshape=(None, *shape))

def read_timestep(group, i):
    timestep = {}
    for key, item in group.items():
      if isinstance(item, h5py.Dataset):
          timestep[key] = item[i]
      elif isinstance(item, h5py.Group):
          timestep[key] = read_timestep(item, i)
    return timestep


def get_stats(group):
    stats_dict = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            stats_dict[key] = {'mean': np.mean(item[:]), 'std': np.std(item[:])}
        elif isinstance(item, h5py.Group):
            stats_dict[key] = get_stats(item)
    return stats_dict

class Recorder:
  def __init__(self,
    name: str,
    observation_space: spaces.Dict,
    action_space: spaces.Dict,
    out_dir: str = 'episodes',
    num_steps: int = 10):
    

    name = os.path.join(out_dir, name)
    if os.path.exists(name + ".hdf5"):
      name = name + datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(Path(name).parent, exist_ok=True)
    
    self.file = h5py.File(name + ".hdf5", "a")

    self.name = name
    os.makedirs(name + "_frames", exist_ok=True)

    self.observation_space = observation_space
    self.action_space = action_space

    self.file.create_group('observation')
    create_dataset_for_space_dict(observation_space, self.file['observation'], num_steps)

    self.file.create_group('action')
    create_dataset_for_space_dict(action_space, self.file['action'], num_steps)

    
    self.index = 0
  
  def record_timestep(self, group: h5py.Group, timestep_dict, i):
    for key, value in timestep_dict.items():
      print('toplevel key: ', key, ', value: ', value)
      if 'bounds' in key:
        continue
      if isinstance(value, dict):
          subgroup = group.require_group(key)
          self.record_timestep(subgroup, value, i)
      else:
          # If the observation is an image (numpy array), save it as an image file
        if isinstance(value, np.ndarray) and len(value.shape) == 3:
            image = Image.fromarray(value)
            image.save(f'{self.name}_frames/{self.index}.png')
        print('key: ', key, ', value: ', value)
        dataset = group[key]
        dataset[i] = value

  def record(self, observation: Any, action: Any):
    self.record_timestep(self.file['observation'], observation, self.index)
    self.record_timestep(self.file['action'], action, self.index)
    self.index += 1
    self.file.attrs['size'] = self.index
  
  def save_stats(self):
    stats = get_stats(self.file)
    print(stats)

    # Add the stats to the hdf5 file
    for key, value in stats.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if np.isscalar(subvalue['mean']):
                    self.file['action'].attrs[f'{key}/{subkey}/mean'] = subvalue['mean']
                    self.file['action'].attrs[f'{key}/{subkey}/std'] = 0
                else:
                    self.file['action'].attrs[f'{key}/{subkey}/mean'] = subvalue['mean']
                    self.file['action'].attrs[f'{key}/{subkey}/std'] = subvalue['std']
        else:
            if np.isscalar(value['mean']):
                self.file['action'].attrs[f'{key}/mean'] = value['mean']
                self.file['action'].attrs[f'{key}/std'] = 0
            else:
                self.file['action'].attrs[f'{key}/mean'] = value['mean']
                self.file['action'].attrs[f'{key}/std'] = value['std']

  def close(self):
    self.file.close()

class Replayer:
  def __init__(self, path: str, observation_space: spaces.Dict, action_space: spaces.Dict):
    '''Replays a dataset from a file. Saves images to folder with _frames appended to the path stem.

    Args:
        path (str): _description_
        observation_space (spaces.Dict): _description_
        action_space (spaces.Dict): _description_
    '''
    self.path = path
    self.frames_path = path.split('.')[0] + '_frames'
    self.file = h5py.File(path, "a")
    self.size = self.file.attrs['size']

    self.observation_space = observation_space
    self.action_space = action_space
    self.index = -1



  def __iter__(self):
    self.index = -1
    return self

  def __next__(self):
    if self.index >= self.size - 1:
      raise StopIteration
    else:
      self.index += 1
      i = self.index

      action = self.action_space.from_jsonable(self.file['action'])[i]
      observation = self.observation_space.from_jsonable(self.file['observation'])[i]

      Image.fromarray(observation['image']).save(f'{self.frames_path}/{i}.png')
      print('i: ', self.index, ', action: ', action)

      return observation, action

  def close(self):
    self.file.close()


