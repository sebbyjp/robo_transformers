# Library for Robotic Transformers. RT-1, RT-X-1, Octo

[![Code Coverage](https://codecov.io/gh/sebbyjp/dgl_ros/branch/code_cov/graph/badge.svg?token=9225d677-c4f2-4607-a9dd-8c22446f13bc)](https://codecov.io/gh/sebbyjp/dgl_ros)
[![ubuntu | python 3.11 | 3.10 | 3.9](https://github.com/sebbyjp/robo_transformers/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/sebbyjp/robo_transformers/actions/workflows/ubuntu.yml)
[![macos | python 3.11 | 3.10 | 3.9](https://github.com/sebbyjp/robo_transformers/actions/workflows/macos.yml/badge.svg)](https://github.com/sebbyjp/robo_transformers/actions/workflows/macos.yml)

## [Available Models](#available-models)

| Model Type |  Variants | Observation Space | Action Space | Author |
| ---------- | --------- | ------- | ------- | ------- |
| [RT-1](https://robotics-transformer1.github.io/)     | rt1main, rt1multirobot, rt1simreal | text + head camera | end effector pose delta |  Google Research, 2022 |
| [RT-1-X](https://robotics-transformer-x.github.io/)  | rt1x   | text + head camera | end effector pose delta |  Google Research et al., 2023 |
| [Octo](https://github.com/octo-models/octo) | octo-base, octo-small | text + head camera + Optional[wrist camera] | end effector pose delta |  Octo Model Team et al., 2023 |


# TODO: Needs to array method, to list method, to torch tensor.

## Installation

Requirements:
python >= 3.9

### From Source

Clone this repo:

`git clone https://github.com/sebbyjp/robo_transformers.git`

Install requirements:

`python -m pip install --upgrade pip`

`cd robo_transformers && pip install -r requirements.txt`

## Run Octo inference on demo images

`python -m robo_transformers.demo`
  
## Run RT-1 Inference On Demo Images

`python -m robo_transformers.models.rt1.inference`

## See usage

You can specify a custom checkpoint path or the model_keys for the three mentioned in the RT-1 paper as well as RT-X.

`python -m robo_transformers.models.rt1.inference --help`

## Run Inference Server

The inference server takes care of all the internal state so all you need to specify is an instruction and image.

```python
from robo_transformers.inference_server import InferenceServer
import numpy as np

# Somewhere in your robot control stack code...

instruction = "pick block"
img = np.random.randn(256, 320, 3) # Width, Height, RGB
inference = InferenceServer()

action = inference(instruction, img)
```

## Data Types

`action, next_policy_state = model.act(time_step, curr_policy_state)`

### policy state is internal state of network

In this case it is a 6-frame window of past observations,actions and the index in time.

```python
{'action_tokens': ArraySpec(shape=(6, 11, 1, 1), dtype=dtype('int32'), name='action_tokens'),
 'image': ArraySpec(shape=(6, 256, 320, 3), dtype=dtype('uint8'), name='image'),
 'step_num': ArraySpec(shape=(1, 1, 1, 1), dtype=dtype('int32'), name='step_num'),
 't': ArraySpec(shape=(1, 1, 1, 1), dtype=dtype('int32'), name='t')}
 ```

### time_step is the input from the environment

```python
{'discount': BoundedArraySpec(shape=(), dtype=dtype('float32'), name='discount', minimum=0.0, maximum=1.0),
 'observation': {'base_pose_tool_reached': ArraySpec(shape=(7,), dtype=dtype('float32'), name='base_pose_tool_reached'),
                 'gripper_closed': ArraySpec(shape=(1,), dtype=dtype('float32'), name='gripper_closed'),
                 'gripper_closedness_commanded': ArraySpec(shape=(1,), dtype=dtype('float32'), name='gripper_closedness_commanded'),
                 'height_to_bottom': ArraySpec(shape=(1,), dtype=dtype('float32'), name='height_to_bottom'),
                 'image': ArraySpec(shape=(256, 320, 3), dtype=dtype('uint8'), name='image'),
                 'natural_language_embedding': ArraySpec(shape=(512,), dtype=dtype('float32'), name='natural_language_embedding'),
                 'natural_language_instruction': ArraySpec(shape=(), dtype=dtype('O'), name='natural_language_instruction'),
                 'orientation_box': ArraySpec(shape=(2, 3), dtype=dtype('float32'), name='orientation_box'),
                 'orientation_start': ArraySpec(shape=(4,), dtype=dtype('float32'), name='orientation_in_camera_space'),
                 'robot_orientation_positions_box': ArraySpec(shape=(3, 3), dtype=dtype('float32'), name='robot_orientation_positions_box'),
                 'rotation_delta_to_go': ArraySpec(shape=(3,), dtype=dtype('float32'), name='rotation_delta_to_go'),
                 'src_rotation': ArraySpec(shape=(4,), dtype=dtype('float32'), name='transform_camera_robot'),
                 'vector_to_go': ArraySpec(shape=(3,), dtype=dtype('float32'), name='vector_to_go'),
                 'workspace_bounds': ArraySpec(shape=(3, 3), dtype=dtype('float32'), name='workspace_bounds')},
 'reward': ArraySpec(shape=(), dtype=dtype('float32'), name='reward'),
 'step_type': ArraySpec(shape=(), dtype=dtype('int32'), name='step_type')}
 ```

### action

```python
{'base_displacement_vector': BoundedArraySpec(shape=(2,), dtype=dtype('float32'), name='base_displacement_vector', minimum=-1.0, maximum=1.0),
 'base_displacement_vertical_rotation': BoundedArraySpec(shape=(1,), dtype=dtype('float32'), name='base_displacement_vertical_rotation', minimum=-3.1415927410125732, maximum=3.1415927410125732),
 'gripper_closedness_action': BoundedArraySpec(shape=(1,), dtype=dtype('float32'), name='gripper_closedness_action', minimum=-1.0, maximum=1.0),
 'rotation_delta': BoundedArraySpec(shape=(3,), dtype=dtype('float32'), name='rotation_delta', minimum=-1.5707963705062866, maximum=1.5707963705062866),
 'terminate_episode': BoundedArraySpec(shape=(3,), dtype=dtype('int32'), name='terminate_episode', minimum=0, maximum=1),
 'world_vector': BoundedArraySpec(shape=(3,), dtype=dtype('float32'), name='world_vector', minimum=-1.0, maximum=1.0)}
 ```

## TODO

- Render action, policy_state, observation specs in something prettier like pandas data frame.
