import unittest 
import os 
import h5py 
import numpy as np 
from gym import spaces
from robo_transformers.data_util import Recorder
class TestRecorder(unittest.TestCase):
    def setUp(self):
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            "instruction": spaces.Discrete(10)
        })
        self.action_space = spaces.Discrete(4)
        self.num_steps = 100
        self.file = h5py.File('test.h5', 'w')
        self.recorder = Recorder(self.file, self.observation_space, self.action_space, self.num_steps)

    def test_record(self):
        observation = {"image": np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8), "instruction": 1}
        action = 2
        self.recorder.record(observation, action)
        self.assertEqual(self.recorder.file['observation/image'].shape, (self.num_steps, 84, 84, 3))
        self.assertEqual(self.recorder.file['observation/instruction'].shape, (self.num_steps,))
        self.assertEqual(self.recorder.file['action'].shape, (self.num_steps,))

    def tearDown(self):
        os.remove('test.h5')

if __name__ == '__main__':
    unittest.main()