from beartype.typing import Any
import h5py
from h5py import string_dtype
import numpy as np
import os
from gym import spaces
from datetime import datetime
from PIL import Image
from pathlib import Path
from absl import logging

def create_dataset_for_space_dict(space_dict: spaces.Dict, group: h5py.Group, num_steps: int = 10):
    logging.debug('toplevel keys: %s', str(space_dict.keys()))
    for key, space in space_dict.items():
        logging.debug(' key: %s, value: %s', key, space)
        if isinstance(space, spaces.Dict):
            subgroup = group.create_group(key)
            create_dataset_for_space_dict(space, subgroup, num_steps)
        else:
            shape = space.shape if space.shape else (num_steps,)
            dtype = space.dtype if space.dtype != str else string_dtype()
            group.create_dataset(key, (num_steps, *shape), dtype=dtype, maxshape=(None, *shape))


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
    out_dir: str = 'episodes'):
    '''Records a dataset to a file. Saves images to folder with _frames appended to the name stem.

    Args:
        name (str): Name of the file.
        observation_space (spaces.Dict): Observation space.
        action_space (spaces.Dict): Action space.
        out_dir (str, optional): Directory of the output file. Defaults to 'episodes'.
    
    Example:
    ```
    # Define the observation and action spaces
    observation_space = spaces.Dict({
        'image': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        'instruction': spaces.Discrete(10)
    })
    action_space = spaces.Dict({
        'gripper_position': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        'gripper_action': spaces.Discrete(2)
    })

    # Create a recorder instance
    recorder = Recorder(name='test_recorder', observation_space=observation_space, action_space=action_space)
    
    # Generate some sample data
    num_steps = 10
    for i in range(num_steps):
        observation = {
            'image': np.ones((224, 224, 3), dtype=np.uint8),
            'instruction': i
        }
        action = {
            'gripper_position': np.zeros((3,), dtype=np.float32),
            'gripper_action': 1
        }
        recorder.record(observation, action)
    
    # Save the statistics
    recorder.save_stats()

    # Close the recorder
    recorder.close()

    # Assert that the HDF5 file and directories are created
    assert os.path.exists('test_recorder.hdf5')
    assert os.path.exists('test_recorder_frames')
    '''

    

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
    create_dataset_for_space_dict(observation_space, self.file['observation'])

    self.file.create_group('action')
    create_dataset_for_space_dict(action_space, self.file['action'])

    
    self.index = 0
  
  def record_timestep(self, group: h5py.Group, timestep_dict, i):
    for key, value in timestep_dict.items():
        logging.debug('toplevel key: %s, value: %s', key, value)
        if 'bounds' in key:
            # Bounds is already present in the space, so we don't need to record it
            continue
        if isinstance(value, dict):
            subgroup = group.require_group(key)
            self.record_timestep(subgroup, value, i)
        else:
            # If the observation is an image (numpy array), save it as an image file
            if isinstance(value, np.ndarray) and len(value.shape) == 3:
                image = Image.fromarray(value)
                image.save(f'{self.name}_frames/{self.index}.png')    
            dataset = group[key]
            dataset[i] = value
        

  def record(self, observation: Any, action: Any):
    if hasattr(observation, 'todict'):
        observation = observation.todict()
    if hasattr(action, 'todict'):
        action = action.todict()
    logging.debug('Recording action: %s for instruction %s', str(action), str(observation.get('instruction', None))) 
    logging.debug('action group keys: %s', str(self.file['action'].keys()))
    self.record_timestep(self.file['observation'], observation, self.index)
    self.record_timestep(self.file['action'], action, self.index)
    
    self.index += 1
    self.file.attrs['size'] = self.index
  
  def save_stats(self):
    stats = get_stats(self.file)
    

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
        path (str): Path to the HDF5 file.
        observation_space (spaces.Dict): Observation space.
        action_space (spaces.Dict): Action space.
    
    Example:
    ```
    # Define the observation and action spaces
    observation_space = spaces.Dict({
        'image': spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        'instruction': spaces.Discrete(10)
    })
    action_space = spaces.Dict({
        'gripper_position': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        'gripper_action': spaces.Discrete(2)
    })

    # Create a replayer instance
    replayer = Replayer(path='test_recorder.hdf5', observation_space=observation_space, action_space=action_space)
    
    # Iterate over the recorded data
    for observation, action in replayer:
        # Perform assertions or additional processing here
        assert isinstance(observation, dict)
        assert isinstance(action, dict)
        assert 'image' in observation
        assert 'gripper_position' in action

        # Save the observation image as a PNG file
        Image.fromarray(observation['image']).save('observation.png')

    # Close the replayer
    replayer.close()
    ```
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
      

      return observation, action

  def close(self):
    self.file.close()


