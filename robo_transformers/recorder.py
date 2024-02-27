import h5py
from h5py import string_dtype
import numpy as np
import os
from gym.spaces import Dict
from robo_transformers.spaces.common import BASIC_VISION_LANGUAGE_OBSERVATION_SPACE, BASIC_BIMANUAL_ACTION_SPACE
from datetime import datetime
from imageio.v3 import imread, imwrite


class Recorder:

  def __init__(self,
               name: str,
               data_dir: str = 'episodes',
               observation_space: Dict = BASIC_VISION_LANGUAGE_OBSERVATION_SPACE,
               action_space: Dict = BASIC_BIMANUAL_ACTION_SPACE,
               num_steps: int = 10):
    if not os.path.exists(data_dir):
      os.makedirs(data_dir)

    name = os.path.join(data_dir, name)
    if os.path.exists(name + ".hdf5"):
      print(f"File {name}.hdf5 already exists. Renaming to {name + datetime.now().strftime('%Y%m%d%H%M%S')}.hdf5")
      os.rename(name + ".hdf5", name + datetime.now().strftime("%Y%m%d%H%M%S") + ".hdf5")
    self.file = h5py.File(name + ".hdf5", "a")

    self.file.create_group('observation')
    for key, space in observation_space.items():
      shape = (*space.shape,) if space.shape is not None else (num_steps,)
      dtype = space.dtype if space.dtype != str else string_dtype()
      self.file['observation'].create_dataset(key, (num_steps, *shape),
                                              dtype=dtype,
                                              maxshape=(None, *shape))

    self.file.create_group('action')
    for key, space_dict in action_space.items():
      self.file['action'].create_group(key)
      for k, space in space_dict.items():
        shape = (*space.shape,) if space.shape is not None else (num_steps,)
        dtype = space.dtype if space.dtype != str else string_dtype()
        self.file[f'action/{key}'].create_dataset(k, (num_steps, *shape),
                                                  dtype=space.dtype,
                                                  maxshape=(None, *shape))

    self.file.create_dataset('reward', (num_steps,), dtype=float, maxshape=(None,))
    self.file.create_dataset('done', (num_steps,), dtype=bool, maxshape=(None,))
    self.index = 0

  def record(self, observation: Dict, action: Dict, reward: float, done: bool):
    for k, v in observation.items():
      if self.index == len(self.file[f'observation/{k}']):
        self.file[f'observation/{k}'].resize(self.index * 2, axis=0)
      self.file[f'observation/{k}'][self.index] = v

    for key, action_dict in action.items():
      for k, v in action_dict.items():
        if self.index == len(self.file[f'action/{key}/{k}']):
          self.file[f'action/{key}/{k}'].resize(self.index * 2, axis=0)
        self.file[f'action/{key}/{k}'][self.index] = v

    if self.index == len(self.file['reward']):
      self.file['reward'].resize(self.index * 2, axis=0)
    self.file['reward'][self.index] = reward

    if self.index == len(self.file['done']):
      self.file['done'].resize(self.index * 2, axis=0)
    self.file['done'][self.index] = done

    self.index += 1
    self.file.attrs['size'] = self.index

  def close(self, save_frames: bool = False, key: str = 'observation/image_primary', save_stats: bool = False):
    if save_frames:
      imwrite(self.file.filename + '_frames.gif', list(self.file[key]))
    if save_stats:
        actions = np.array([
            np.array([
                self.recorder.file["action/left_hand/x"][i],
                self.recorder.file["action/left_hand/y"][i],
                self.recorder.file["action/left_hand/z"][i],
                self.recorder.file["action/left_hand/roll"][i],
                self.recorder.file["action/left_hand/pitch"][i],
                self.recorder.file["action/left_hand/yaw"][i],
                self.recorder.file["action/left_hand/grasp"][i],
            ]) for i in range(self.file.attrs["size"])
        ])

        self.file.attrs["mean"] = np.mean(actions, axis=0)
        self.file.attrs["std"] = np.std(actions, axis=0)

    self.file.close()
