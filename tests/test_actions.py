import unittest
from gym.spaces import Dict, Box
from robo_transformers.common.actions import PoseControl, PlanarDirectionControl, JointControl, GripperControl, FullJointControl, EmbodiedControl, GripperBaseControl



class TestPoseControl(unittest.TestCase):

  def test_space(self):
    pose_control = PoseControl()
    space = pose_control.space()
    self.assertIsInstance(space, Dict)
    self.assertIsInstance(space['xyz'], Box)
    self.assertIsInstance(space['rpy'], Box)
    self.assertEqual(space['xyz'].low, -1.0)
    self.assertEqual(space['xyz'].high, 1.0)
    self.assertEqual(space['rpy'].low, -3.14)
    self.assertEqual(space['rpy'].high, 3.14)


class TestPlanarDirectionControl(unittest.TestCase):

  def test_space(self):
    planar_direction_control = PlanarDirectionControl()
    space = planar_direction_control.space()
    self.assertIsInstance(space, Dict)
    self.assertIsInstance(space['xy'], Box)
    self.assertIsInstance(space['yaw'], Box)


class TestJointControl(unittest.TestCase):

  def test_space(self):
    joint_control = JointControl()
    space = joint_control.space()
    self.assertIsInstance(space, Dict)
    self.assertIsInstance(space['value'], Box)
    self.assertEqual(space['value'].low, -3.14)
    self.assertEqual(space['value'].high, 3.14)


class TestGripperControl(unittest.TestCase):

  def test_space(self):
    gripper_control = GripperControl()
    space = gripper_control.space()
    self.assertIsInstance(space, Dict)
    self.assertIsInstance(space['pose'], Dict)
    self.assertIsInstance(space['grasp'], Dict)
    self.assertIsInstance(space['pose']['xyz'], Box)
    self.assertIsInstance(space['pose']['rpy'], Box)
    self.assertIsInstance(space['grasp']['value'], Box)
    self.assertEqual(space['pose']['xyz'].low, -1.0)
    self.assertEqual(space['pose']['xyz'].high, 1.0)
    self.assertEqual(space['pose']['rpy'].low, -3.14)
    self.assertEqual(space['pose']['rpy'].high, 3.14)
    self.assertEqual(space['grasp']['value'].low, -3.14)
    self.assertEqual(space['grasp']['value'].high, 3.14)


class TestFullJointControl(unittest.TestCase):

  def test_space(self):
    full_joint_control = FullJointControl()
    space = full_joint_control.space()
    self.assertIsInstance(space, Dict)
    self.assertIsInstance(space['joints'], Box)
    self.assertEqual(space['joints'].low, -3.14)
    self.assertEqual(space['joints'].high, 3.14)


class TestEmbodiedControl(unittest.TestCase):

  def test_space(self):
    embodied_control = EmbodiedControl()
    space = embodied_control.space()
    self.assertIsInstance(space, Dict)
    self.assertIsInstance(space['actions'], Box)
    self.assertEqual(space['actions'].low, -3.14)
    self.assertEqual(space['actions'].high, 3.14)


class TestGripperBaseControl(unittest.TestCase):

  def test_space(self):
    gripper_base_control = GripperBaseControl()
    space = gripper_base_control.space()
    self.assertIsInstance(space, Dict)
    self.assertIsInstance(space['base'], Dict)
    self.assertIsInstance(space['left_gripper'], Dict)
    self.assertIsInstance(space['base']['xy'], Box)
    self.assertIsInstance(space['base']['yaw'], Box)
    self.assertIsInstance(space['left_gripper']['pose'], Dict)
    self.assertIsInstance(space['left_gripper']['grasp'], Dict)
    self.assertIsInstance(space['left_gripper']['pose']['xyz'], Box)
    self.assertIsInstance(space['left_gripper']['pose']['rpy'], Box)
    self.assertIsInstance(space['left_gripper']['grasp']['value'], Box)
    self.assertEqual(space['base']['xy'].low, -1.0)
    self.assertEqual(space['base']['xy'].high, 1.0)
    self.assertEqual(space['base']['yaw'].low, -3.14)
    self.assertEqual(space['base']['yaw'].high, 3.14)
    self.assertEqual(space['left_gripper']['pose']['xyz'].low, -1.0)
    self.assertEqual(space['left_gripper']['pose']['xyz'].high, 1.0)
    self.assertEqual(space['left_gripper']['pose']['rpy'].low, -3.14)
    self.assertEqual(space['left_gripper']['pose']['rpy'].high, 3.14)
    self.assertEqual(space['left_gripper']['grasp']['value'].low, -3.14)
    self.assertEqual(space['left_gripper']['grasp']['value'].high, 3.14)


if __name__ == '__main__':
  unittest.main()