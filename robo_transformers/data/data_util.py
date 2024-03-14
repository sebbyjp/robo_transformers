from beartype.typing import Any, Callable
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
  print('data group keys:', str(space_dict.keys()))
  for key, space in space_dict.items():
    logging.debug(' key: , value: %s', key, space)
    if isinstance(space, spaces.Dict):
      subgroup = group.create_group(key)
      create_dataset_for_space_dict(space, subgroup, num_steps)
    else:
      shape = space.shape if hasattr(space, 'shape') and space.shape is not None else (num_steps,)
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
               out_dir: str = 'episodes',
               num_steps: int = 10,
               image_keys_to_save: list = ['image']):
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
    self.num_steps = num_steps
    self.image_keys_to_save = image_keys_to_save

    self.file.create_group('observation')
    create_dataset_for_space_dict(observation_space, self.file['observation'], self.num_steps)

    self.file.create_group('action')
    create_dataset_for_space_dict(action_space, self.file['action'], self.num_steps)

    self.index = 0

  def record_timestep(self, group: h5py.Group, timestep_dict, i):
    logging.debug('group keys: %s', str(group.keys()))
    for key, value in timestep_dict.items():
      logging.debug(' key: %s, value: %s', key, value)
      if 'bounds' in key:
        # Bounds is already present in the space, so we don't need to record it
        continue
      if isinstance(value, dict):
        subgroup = group.require_group(key)
        self.record_timestep(subgroup, value, i)
      else:
        # If the observation is an image (numpy array), save it as an image file
        if isinstance(value, np.ndarray) and len(value.shape) == 3 and key in self.image_keys_to_save:
          image = Image.fromarray(value)
          image.save(f'{self.name}_frames/{self.index}.png')
        logging.debug(' about to record key: %s, value: %s', key, value)
        dataset = group[key]
        if i >= dataset.shape[0]:
            dataset.resize((2*i, *dataset.shape[1:]))
    
        dataset[i] = value

  def record(self, observation: Any, action: Any):
    if hasattr(observation, 'todict'):
      observation = observation.todict()
    if hasattr(action, 'todict'):
      action = action.todict()
    logging.debug('Recording action: %s for instruction %s', str(action),
                  str(observation.get('instruction', None)))

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

    def __init__(self, path: str, image_keys_to_save: list = []):
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
        self.frames_path = Path(self.path).stem + '_frames'
        self.file = h5py.File(path, "a")
        self.size = self.file.attrs['size']
        self.image_keys_to_save = image_keys_to_save
        self.index = -1

    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        if self.index >= self.size - 1:
            raise StopIteration
        else:
            self.index += 1
        if len(self.image_keys_to_save) > 0:
            os.makedirs(self.frames_path, exist_ok=True)
        return self.read_sample(self.index, self.image_keys_to_save)

    def read_sample(self, index:int, image_keys_to_save: list = []):
        '''Reads a sample from the dataset. 
            TODO(sebbyj): Add support for reading a range of samples.
            TODO(sebbyj): Add support for reading a range of keys.

        Args:
            index (int): _description_
            image_keys_to_save (list, optional): _description_. Defaults to [].
        '''
        def recursive_read(group, index):
            result = {}
            for key, value in group.items():
                if isinstance(value, h5py.Group):
                    result[key] = recursive_read(value, index)
                else:
                    if key in image_keys_to_save and len(value.shape) == 3:
                        Image.fromarray(value).save(f'{self.frames_path}/{key}_{index}.png')
                    result[key] = value[index]
            return result

        observation = recursive_read(self.file['observation'], index)
        action = recursive_read(self.file['action'], index)

        return observation, action
    
    def close(self):
        self.file.close()



def transform_datasets(filenames: list, on_keys: list, transforms: list, out_dir: str = 'transformed_datasets'):
    from robo_transformers.common.observations import ImageInstruction, Image as MbdImage
    from robo_transformers.interface import Control
    from robo_transformers.common.actions import GripperBaseControl, JointControl, PlanarDirectionControl, PoseControl, GripperControl
    recorder = Recorder(
          Path('concat').stem, ImageInstruction(image=MbdImage(224,224)).space(),
          GripperBaseControl(
              camera_pan=JointControl(),
              camera_tilt=JointControl(),
          ).space(), out_dir, 500)
    for f in filenames:
        replayer = Replayer(f)
      
        for i, item in enumerate(replayer):

            observation, action = item
            action = GripperBaseControl(
                base=PlanarDirectionControl(Control.RELATIVE, action['base']['xy'], action['base']['yaw']),
                camera_pan=JointControl(Control.RELATIVE, action['camera_pan']['value']),
                camera_tilt=JointControl(Control.RELATIVE, action['camera_tilt']['value']),
                left_gripper=GripperControl(Control.UNSPECIFIED, PoseControl(Control.RELATIVE, action['left_gripper']['pose']['xyz'], action['left_gripper']['pose']['rpy']), JointControl(Control.ABSOLUTE, action['left_gripper']['grasp']['value'])
                ))
          
                
            
            print('action:', action)
            observation = ImageInstruction(image=np.array(PILImage.fromarray(observation['image']).resize((224,224))),
            instruction=observation['instruction'])
            recorder.record(observation, action)
        
        recorder.save_stats()
        recorder.close()

if __name__ == '__main__':
  from PIL import Image as PILImage
  from robo_transformers.common.observations import ImageInstruction
  from robo_transformers.interface import Control
  from robo_transformers.common.actions import GripperBaseControl, JointControl, PlanarDirectionControl, PoseControl, GripperControl
  # print(GripperBaseControl(
  #               camera_pan=JointControl(),
  #               camera_tilt=JointControl(),
  #           ).left_gripper.flatten())
  transform_datasets([
  'transformed_datasets/set_table.hdf5'], ['observation/image', ], [], 'transformed_datasets')

