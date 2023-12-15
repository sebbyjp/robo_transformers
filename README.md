[![Code Coverage](https://codecov.io/gh/sebbyjp/dgl_ros/branch/code_cov/graph/badge.svg?token=9225d677-c4f2-4607-a9dd-8c22446f13bc)](https://codecov.io/gh/sebbyjp/dgl_ros)

# Library for Robotic Transformers. RT-1 and RT-X-1.

## Install:

Requirements:
- python3.11

Clone this repo: 
`git clone https://github.com/sebbyjp/robo_transformers.git --recurse-submodules`

`cd robo_transformers`

Use poetry

`pip install poetry && poetry config virtualenvs.in-project true`

**Install dependencies:**
`poetry install`
`source .venv/bin/activate`
  
## Run Inference On Demo Images.
`python robo_transformers/inference/rt1/rt1_inference.py`
  
# Downloading checkpoints.
To install the checkpoints from the robotics_transformer git repo, you will need git-lfs
- Install git-lfs (use brew or apt if on unix), then

`cd third_party/rt1/robotics_transformer && git lfs install` 
`git lfs pull https://www.github.com/google-research/robotics_transformer.git `

### Optional: Download RT-1-X model from the Open-X Embodiment paper.
- Install gsutil: `pip install gsutil`
- Run: `gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip ./checkpoints/`
- Unzip: `cd checkpoints && unzip rt_1_x_tf_trained_for_002272480_step.zip`
  
## Notes
`action, next_policy_state = model.act(time_step, curr_policy_state)`
### policy state is internal state of network:
In this case it is a 6-frame window of past observations,actions and the index in time.
```
{'action_tokens': ArraySpec(shape=(6, 11, 1, 1), dtype=dtype('int32'), name='action_tokens'),
 'image': ArraySpec(shape=(6, 256, 320, 3), dtype=dtype('uint8'), name='image'),
 'step_num': ArraySpec(shape=(1, 1, 1, 1), dtype=dtype('int32'), name='step_num'),
 't': ArraySpec(shape=(1, 1, 1, 1), dtype=dtype('int32'), name='t')}
 ```


### time_step is the input from the environment:
```
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

### action:
```
{'base_displacement_vector': BoundedArraySpec(shape=(2,), dtype=dtype('float32'), name='base_displacement_vector', minimum=-1.0, maximum=1.0),
 'base_displacement_vertical_rotation': BoundedArraySpec(shape=(1,), dtype=dtype('float32'), name='base_displacement_vertical_rotation', minimum=-3.1415927410125732, maximum=3.1415927410125732),
 'gripper_closedness_action': BoundedArraySpec(shape=(1,), dtype=dtype('float32'), name='gripper_closedness_action', minimum=-1.0, maximum=1.0),
 'rotation_delta': BoundedArraySpec(shape=(3,), dtype=dtype('float32'), name='rotation_delta', minimum=-1.5707963705062866, maximum=1.5707963705062866),
 'terminate_episode': BoundedArraySpec(shape=(3,), dtype=dtype('int32'), name='terminate_episode', minimum=0, maximum=1),
 'world_vector': BoundedArraySpec(shape=(3,), dtype=dtype('float32'), name='world_vector', minimum=-1.0, maximum=1.0)}
 ```

 ## TODO:
 - Render action, policy_state, observation specs in something prettier like pandas data frame.