from robo_transformers.replayer import Replayer
import os
from PIL import Image
def test_replayer():
  replay = Replayer('episodes_best/pick_coke_can.hdf5')
  for i, timestep in enumerate(replay):
    assert len(timestep) == 4
    observation, action, reward, done = timestep
    assert observation.keys() == {'image_primary', 'image_secondary','language_instruction'}
    assert observation['image_primary'].shape == (224, 224, 3)
    assert observation['image_secondary'].shape == (128, 128, 3)
    assert action.keys() == {'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grasp'}

  replay.close(save_frames=True, print_stats=True, save_stats=True)
  assert os.path.exists('episodes_best/pick_coke_can.gif')

if __name__ == '__main__':
  test_replayer()