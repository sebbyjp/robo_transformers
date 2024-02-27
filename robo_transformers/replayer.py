from einops import rearrange
from imageio.v3 import imread, imwrite
import h5py
from pathlib import Path
import os
from gym.spaces import Dict
from robo_transformers.spaces.common import BASIC_VISION_LANGUAGE_OBSERVATION_SPACE, BASIC_BIMANUAL_ACTION_SPACE
import numpy as np
from pprint import pprint
from typing import Dict as TypedDict

class Replayer:
  def __init__(self, path,  observation_space: Dict = BASIC_VISION_LANGUAGE_OBSERVATION_SPACE, action_space: Dict = BASIC_BIMANUAL_ACTION_SPACE):
    self.file = h5py.File(path, "a")
    self.size = self.file.attrs['size']

    self.observation_space = observation_space
    self.action_space = action_space
    self.index = -1


  def stats(self):
    return {
      'size': self.size,
      'observation_space': self.observation_space,
      'action_space': self.action_space,
      'mean': self.file.attrs['mean'],
      'std': self.file.attrs['std'],
    }
  def __iter__(self):
    self.index = -1
    return self

  def __next__(self):
    if self.index >= self.size:
      raise StopIteration
    else:
      self.index += 1
      i = self.index

      # observation = apply_fn(self.observation_space, lambda k: self.file[k][i], prefix='observation')
      # action = apply_fn(self.action_space, lambda k: self.file[k][i], prefix='action')
      # reward = self.file['reward'][i]
      # done = self.file['done'][i]
      observation = {
        'image_primary': self.file['observation/image_primary'][i],
        'image_secondary': self.file['observation/image_secondary'][i],
        'language_instruction': self.file['observation/language_instruction'][i],
      }
      action = {
              'x': self.file['action/left_hand/x'][i],
              'y': self.file['action/left_hand/y'][i],
              'z': self.file['action/left_hand/z'][i],
              'roll': self.file['action/left_hand/roll'][i],
              'pitch': self.file['action/left_hand/pitch'][i],
              'yaw': self.file['action/left_hand/yaw'][i],
              'grasp': self.file['action/left_hand/grasp'][i],

      }
      return  observation, action, self.file['reward'][i], self.file['done'][i]
  
  def save_frames(self, index=0, stop_index=-1, key='observation/image_primary'):
    if stop_index == -1:
      stop_index = self.size
    
    frames = np.stack([self.file[key][i] for i in range(index, stop_index)])
    path = Path(self.file.filename)
    imwrite(path.with_suffix('.gif'), frames)




  def close(self, save_frames: bool = False, index=0, stop_index=-1, key: str = 'observation/image_primary', print_stats: bool = True, save_stats: bool = True):
    if save_frames:
      self.save_frames(index, stop_index, key)
    if save_stats:
        actions = np.array([
            np.array([
                self.file["action/left_hand/x"][i],
                self.file["action/left_hand/y"][i],
                self.file["action/left_hand/z"][i],
                self.file["action/left_hand/roll"][i],
                self.file["action/left_hand/pitch"][i],
                self.file["action/left_hand/yaw"][i],
                self.file["action/left_hand/grasp"][i],
            ]) for i in range(self.file.attrs["size"])
        ])

        self.file.attrs["mean"] = np.mean(actions, axis=0)
        self.file.attrs["std"] = np.std(actions, axis=0)
    if print_stats:
      pprint(self.stats())
    self.file.close()


#




# import numpy as np
# import tensorflow as tf
# from oxe_envlogger.data_type import get_gym_space
# from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType


# class Recorder:
#     def __init__(self):
#         obs_sample = { 'end_effector_cartesian_pos': tf.zeros(shape=(7,), dtype=tf.float32),
#             'end_effector_cartesian_velocity': tf.zeros(shape=(6,), dtype=tf.float32),
#             'image': tf.zeros(shape=(224, 224, 3), dtype=tf.uint8),
#             'image_wrist': tf.zeros(shape=(224, 224, 3), dtype=tf.uint8),
#             'joint_pos': tf.zeros(shape=(8,), dtype=tf.float32),
#             'natural_language_instruction': " asdf"}

#         action_sample =  {'gripper_closedness_action': tf.zeros(shape=(1,), dtype=tf.float32),
#             'terminate_episode': tf.zeros(shape=(3,), dtype=tf.int32),
#             'world_vector': tf.zeros(shape=(3,), dtype=tf.float32),}
#         self.logger = RLDSLogger(
#         observation_space=get_gym_space(obs_sample),
#         action_space=get_gym_space(action_sample),
#         dataset_name="test",
#         directory="logs",
#         max_episodes_per_file=1,
# )


#     def record(self, observation, action, reward, done, info):
#        # 0. sample data
#         obs_sample = { 'end_effector_cartesian_pos': tf.zeros(shape=(7,), dtype=tf.float32),
#             'end_effector_cartesian_velocity': tf.zeros(shape=(6,), dtype=tf.float32),
#             'image': tf.zeros(shape=(224, 224, 3), dtype=tf.uint8),
#             'image_wrist': tf.zeros(shape=(224, 224, 3), dtype=tf.uint8),
#             'joint_pos': tf.zeros(shape=(8,), dtype=tf.float32),
#             'natural_language_instruction': " asdf"}

#         action_sample =  {'gripper_closedness_action': tf.zeros(shape=(1,), dtype=tf.float32),
#             'terminate_episode': tf.zeros(shape=(3,), dtype=tf.int32),
#             'world_vector': tf.zeros(shape=(3,), dtype=tf.float32),}

#         # 2. log data
#         self.logger(action_sample, obs_sample, 1.0, step_type=RLDSStepType.RESTART)
#         self.logger(action_sample, obs_sample, 1.0)
#         self.logger(action_sample, obs_sample, 1.0, step_type=RLDSStepType.TERMINATION)
#         self.logger.close() # this is important to flush the current data to disk
