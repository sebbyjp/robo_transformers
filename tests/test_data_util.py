import os
import numpy as np
from gym import spaces
from PIL import Image
from robo_transformers.data_util import Recorder, Replayer

def test_recorder():
    # Create a temporary directory for testing
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)

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
    recorder = Recorder(name='test_recorder', out_dir=temp_dir, observation_space=observation_space, action_space=action_space)

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
    assert os.path.exists(os.path.join(temp_dir, 'test_recorder.hdf5'))
    assert os.path.exists(os.path.join(temp_dir, 'test_recorder_frames'))
    assert os.path.exists(os.path.join(temp_dir, 'test_recorder_frames', '0.png'))

     # Create a replayer instance
    replayer = Replayer(path=os.path.join(temp_dir, 'test_recorder.hdf5'), observation_space=observation_space, action_space=action_space)

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

    # Assert that the observation image file is created
    assert os.path.exists('observation.png')

    # Clean up the observation image file
    os.remove('observation.png')

    # Clean up the temporary directory
    os.remove(os.path.join(temp_dir, 'test_recorder.hdf5'))
    for file in os.listdir(os.path.join(temp_dir, 'test_recorder_frames')):
        os.remove(os.path.join(temp_dir, 'test_recorder_frames', file))
    os.rmdir(os.path.join(temp_dir, 'test_recorder_frames'))
    os.rmdir(temp_dir)


   

# Run the tests
test_recorder()