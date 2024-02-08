
import h5py
from h5py import string_dtype
import numpy as np
import os
from gym.spaces import Dict
from robo_transformers.spaces.common import EEF_ACTION_SPACE, BASIC_VISION_LANGUAGE_OBSERVATION_SPACE
from robo_transformers.spaces.util import apply_fn



class Recorder:
    def __init__(self, name: str, data_dir: str = 'episodes', observation_space: Dict = BASIC_VISION_LANGUAGE_OBSERVATION_SPACE, action_space: Dict = EEF_ACTION_SPACE, num_steps: int = 10):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        name = os.path.join(data_dir, name)
        self.file = h5py.File(name + ".hdf5", "a")

        self.file.create_group('observation')
        for key, space in observation_space.items():
            shape = (*space.shape,) if space.shape is not None else (num_steps,)
            dtype = space.dtype if space.dtype != str else string_dtype()
            self.file['observation'].create_dataset(key, (num_steps,*shape), dtype=dtype, maxshape=(None, *shape))
        
        self.file.create_group('action')
        for key, space_dict in action_space.items():
            self.file['action'].create_group(key)
            for k, space in space_dict.items():
                shape = (*space.shape,) if space.shape is not None else (num_steps,)
                dtype = space.dtype if space.dtype != str else string_dtype()
                self.file[f'action/{key}'].create_dataset(k, (num_steps,*shape), dtype=space.dtype, maxshape=(None, *shape))

        self.file.create_dataset('reward', (num_steps,), dtype=float, maxshape=(None,))
        self.file.create_dataset('done', (num_steps,), dtype=bool, maxshape=(None,))
        self.index = 0


    def record(self, observation: Dict, action: Dict, reward: float, done: bool):
        for k, v in observation.items():
            if self.index == len(self.file[f'observation/{k}']):
                self.file[f'observation/{k}'].resize(self.index*2, axis=0)
            self.file[f'observation/{k}'][self.index] = v
        
        for key, action_dict in action.items():
            for k, v in action_dict.items():
                if self.index == len(self.file[f'action/{key}/{k}']): 
                    self.file[f'action/{key}/{k}'].resize(self.index*2, axis=0)
                self.file[f'action/{key}/{k}'][self.index] = v

        if self.index == len(self.file['reward']): 
                self.file['reward'].resize(self.index*2, axis=0)
        self.file['reward'][self.index] = reward

        if self.index == len( self.file['done']):
            self.file['done'].resize(self.index*2, axis=0)
        self.file['done'][self.index] = done

        self.index += 1
        self.file.attrs['size'] = self.index
    
    def close(self):
        self.file.close()

class Replayer:
    def __init__(self, name: str, data_dir: str = 'episodes', observation_space: Dict = BASIC_VISION_LANGUAGE_OBSERVATION_SPACE, action_space: Dict = EEF_ACTION_SPACE):
        name = os.path.join(data_dir, name)
        self.file = h5py.File(name + ".hdf5", "r")
        self.size = self.file.attrs['size']

        self.observation_space = observation_space
        self.action_space = action_space
        self.index = -1



    def __iter__(self):
        self.index = -1
        return self

    def __next__(self):
        if self.index >= self.size:
            raise StopIteration
        else:
            self.index += 1
            i = self.index

            observation = apply_fn(self.observation_space, lambda k: self.file[k][i], prefix='observation')
            action = apply_fn(self.action_space, lambda k: self.file[k][i], prefix='action')
            reward = self.file['reward'][i]
            done = self.file['done'][i]

            return observation, action, reward, done



    def close(self):
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



