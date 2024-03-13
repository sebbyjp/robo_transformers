from dataclasses import dataclass, field
from avp_stream import VisionProStreamer
from robo_transformers.interface import Agent, Control, ControlAction
from robo_transformers.data_util import Recorder
from robo_transformers.common.observations import ImageInstruction
from robo_transformers.common.actions import GripperBaseControl, GripperControl, JointControl, PoseControl, PlanarDirectionControl
from beartype.typing import Any, Optional
import os
import cv2
import numpy as np
import numpy.typing as npt
from beartype import beartype
import math
from gym import spaces
from scipy.spatial.transform import Rotation as R
from agents_corp.rt_genie.smart_vlacm import SVLACM
from agents_corp.rt_genie.motion_control import MotionController, LocobotJoyController, LocobotControl
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def matrix_to_xyz_rpy(matrix):
    '''Convert a 4x4 matrix to xyz and rpy.'''
    matrix = np.squeeze(matrix)
    assert matrix.shape == (4, 4)
    # Extract translation
    x, y, z = matrix[:3, 3]
    # Extract rotation matrix and convert to euler angles
    r = R.from_matrix(matrix[:3, :3])
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)
    return x, y, z, roll, pitch, yaw

def xyz_rpy_to_matrix(x, y, z, roll, pitch, yaw):
    # Convert euler angles to rotation matrix
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    # Create a 4x4 matrix
    matrix = np.eye(4)
    # Insert rotation matrix
    matrix[:3, :3] = r.as_matrix()
    # Insert translation
    matrix[:3, 3] = [x, y, z]
    return matrix


def get_transformation(matrix1, matrix2):
    '''Compute the transformation from matrix1 to matrix2.'''
    # Ensure matrices are numpy arrays
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    # Compute the inverse of matrix1
    matrix1_inv = np.linalg.inv(matrix1)

    # Compute the transformation from matrix1 to matrix2
    transformation = np.dot(matrix2, matrix1_inv)

    return transformation

@dataclass
class HumanoidControl(ControlAction):
    head: np.ndarray = np.zeros((4, 4))
    right_wrist: np.ndarray = np.zeros((4, 4))
    left_wrist: np.ndarray = np.zeros((4, 4))
    # right_fingers: np.ndarray = np.zeros((25, 4, 4))
    # left_fingers: np.ndarray = np.zeros((25, 4, 4))
    right_pinch_distance: JointControl = field(default_factory=lambda: JointControl())
    left_pinch_distance: JointControl = field(default_factory=lambda: JointControl())
    right_wrist_roll: JointControl = field(default_factory=lambda: JointControl())
    left_wrist_roll: JointControl = field(default_factory=lambda: JointControl())

    # left_wrist: np.ndarray =  np.eye(4)
    # right_fingers: np.ndarray = np.eye(4)
    # left_fingers: np.ndarray = np.eye(4)
    # right_pinch_distance: float = 0
    # left_pinch_distance: float = 0
    # right_wrist_roll:np.ndarray = np.eye(4)
    # left_wrist_roll:np.ndarray = np.eye(4)


    def get_relative(self, other: 'HumanoidControl'):
        """Get the relative transformation to another HumanoidControl."""
        relative = HumanoidControl(
            head=get_transformation(other.head, self.head),
            right_wrist=get_transformation(other.right_wrist, self.right_wrist),
            left_wrist=get_transformation(other.left_wrist, self.left_wrist),
            # right_fingers=get_transformation(other.right_fingers, self.right_fingers),
            # left_fingers=get_transformation(other.left_fingers, self.left_fingers),
            right_pinch_distance=self.right_pinch_distance.value - other.right_pinch_distance.value,
            left_pinch_distance=self.left_pinch_distance.value - other.left_pinch_distance.value,
            right_wrist_roll=self.right_wrist_roll.value - other.right_wrist_roll.value,
        )
        return relative


def humanoid_to_gripper_base_control(humanoid_control: HumanoidControl) -> GripperBaseControl:
    ''' Calculate the desired position of the gripper base

        Args:
            head (np.ndarray): shape (1,4,4) / measured from ground frame
            right_wrist (np.ndarray): shape (1,4,4) / measured from ground frame
            left_wrist (np.ndarray): shape (1,4,4) / measured from ground frame
            right_pinch_distance (float): distance between right index tip and thumb tip 
            left_pinch_distance (float): distance between left index tip and thumb tip 
            right_wrist_roll (float): rotation angle of your right wrist around your arm axis
            left_wrist_roll (float): rotation angle of your left wrist around your arm axis

        Returns:
            GripperBaseControl: The desired position of the gripper base

        Example Output:
        {'control_type': <Control.UNSPECIFIED: 0>, 'base': {'control_type': <Control.RELATIVE: 2>, 'xy': [0, 0], 'yaw': [0], 'xy_bounds': [-1, 1], 'yaw_bounds': [-3.14, 3.14]}, 'left_gripper': {'control_type': <Control.UNSPECIFIED: 0>, 'pose': {'control_type': <Control.UNSPECIFIED: 0>, 'xyz': [0, 0, 0], 'rpy': [0, 0, 0], 'xyz_bounds': [-1, 1], 'rpy_bounds': [-3.14, 3.14]}, 'grasp': {'control_type': <Control.ABSOLUTE: 1>, 'value': 0, 'bounds': [0, 1]}}, 'finish': False, 'action_time': 0.2}
    '''
    # Calculate the desired planar pose of the base
    head = humanoid_control.head
    right_wrist = humanoid_control.right_wrist
    left_wrist = humanoid_control.left_wrist
    right_pinch_distance = humanoid_control.right_pinch_distance
    left_pinch_distance = humanoid_control.left_pinch_distance

    base_x, base_y, _, _, base_pitch, base_yaw = matrix_to_xyz_rpy(head)
    
    return GripperBaseControl(
        base=PlanarDirectionControl(
            control_type=Control.RELATIVE,
            xy=[base_x, base_y],
            yaw=base_yaw,
            xy_bounds=[-1, 1],
            yaw_bounds=[-2*math.pi, 2*math.pi]
        ),
        left_gripper=GripperControl(
            pose=PoseControl(
                control_type=Control.RELATIVE,
                xyz=matrix_to_xyz_rpy(left_wrist)[:3],
                rpy=matrix_to_xyz_rpy(left_wrist)[3:],
                xyz_bounds=[-1, 1],
                rpy_bounds=[-math.pi, math.pi]
            ),
            grasp=JointControl(control_type=Control.ABSOLUTE, value=left_pinch_distance, bounds=[0, 1]),
        ),
        # right_gripper=GripperControl(
        #     pose=PoseControl(
        #         control_type=Control.RELATIVE,
        #         xyz=matrix_to_xyz_rpy(right_wrist)[:3],
        #         rpy=matrix_to_xyz_rpy(right_wrist)[3:],
        #         xyz_bounds=[-1, 1],
        #         rpy_bounds=[-math.pi, math.pi]
        #     ),
        #     grasp=JointControl(control_type=Control.ABSOLUTE, value=right_pinch_distance, bounds=[0, 1]),
        # ),
        finish=False,
        camera_pan=JointControl(Control.RELATIVE, value=base_yaw, bounds=[-math.pi, math.pi]),
        camera_tilt=JointControl(Control.RELATIVE, value=base_pitch, bounds=[-math.pi, math.pi]),
    )
   
   

'''
Example:
from avp_stream import VisionProStreamer
avp_ip = "10.31.181.201"   # example IP 
s = VisionProStreamer(ip = avp_ip, record = True)

while True:
    r = s.latest
    print(r['head'], r['right_wrist'], r['right_fingers'])
# r = s.latest
# r is a dictionary containing the following data streamed from AVP:
r['head']: np.ndarray  
  # shape (1,4,4) / measured from ground frame
r['right_wrist']: np.ndarray 
  # shape (1,4,4) / measured from ground frame
r['left_wrist']: np.ndarray 
  # shape (1,4,4) / measured from ground frame
r['right_fingers']: np.ndarray 
  # shape (25,4,4) / measured from right wrist frame 
r['left_fingers']: np.ndarray 
  # shape (25,4,4) / measured from left wrist frame 
r['right_pinch_distance']: float  
  # distance between right index tip and thumb tip 
r['left_pinch_distance']: float  
  # distance between left index tip and thumb tip 
r['right_wrist_roll']: float 
  # rotation angle of your right wrist around your arm axis
r['left_wrist_roll']: float 
 # rotation angle of your left wrist around your arm axis
'''



@beartype
class VisionProAgent(Agent):

  def __init__(
      self,
      name: str,
      device_ip: str, 
      data_dir: str = "episodes",
      observation_space: Optional[spaces.Space] = None,
      action_space: Optional[spaces.Space] = None,
      **kwargs
  ) -> None:
    ''' Agent for VisionPro Teleop Agent

        Args:
            name (str): The name of the hdf5 file.
            device_ip (str): The IP of the device.
            data_dir (str, optional): The directory to save the data. Defaults to "episodes".
            xyz_step (float, optional): Step size for xyz. Defaults to 0.02.
            rpy_step (float, optional): Step size for rpy. Defaults to PI/8.
       
    '''
    self.action_space = action_space
    self.observation_space = observation_space

    if self.action_space is None:
        self.action_space = LocobotControl().space()
    if self.observation_space is None:
        self.observation_space = ImageInstruction().space()

    self.humanoid_control = None
    self.recorder = Recorder(name,out_dir=data_dir, observation_space=self.observation_space, action_space=self.action_space, **kwargs)
    self.last_grasp = 0
    self.visionpro_streamer = VisionProStreamer(device_ip)

  def act(
      self,
      instruction: str,
      image: npt.ArrayLike,
  ) -> list:
    # Create observation of past `window_size` number of observations
    r = self.visionpro_streamer.latest
    absolute_humanoid_control =   HumanoidControl(
            head=r['head'],
            right_wrist=r['right_wrist'],
            left_wrist=r['left_wrist'],
            # right_fingers=r['right_fingers'],
            # left_fingers=r['left_fingers'],
            right_pinch_distance=JointControl(control_type=Control.ABSOLUTE, value=r['right_pinch_distance']),
            left_pinch_distance=JointControl(control_type=Control.ABSOLUTE, value=r['left_pinch_distance']),
            right_wrist_roll=JointControl(control_type=Control.ABSOLUTE, value=r['right_wrist_roll']),
            left_wrist_roll=JointControl(control_type=Control.ABSOLUTE, value=r['left_wrist_roll'])


        )
    
    if self.humanoid_control is None:
        self.humanoid_control = absolute_humanoid_control
    
    relative_humanoid_control = absolute_humanoid_control.get_relative(self.humanoid_control)
    self.humanoid_control = absolute_humanoid_control

    gripper_base_control = humanoid_to_gripper_base_control(relative_humanoid_control)
    print('gripper x,y,z,roll,pitch,yaw:', gripper_base_control.left_gripper.pose.xyz, gripper_base_control.left_gripper.pose.rpy)
    print('gripper grasp:', gripper_base_control.left_gripper.grasp.value)
    print('gripper base x,y,yaw:', gripper_base_control.base.xy, gripper_base_control.base.yaw)
    return [humanoid_to_gripper_base_control(relative_humanoid_control).todict()]


if __name__ == "__main__":
    agent = VisionProAgent("test", device_ip="10.4.33.50")
    while True:
        agent.act("", np.zeros((480, 640, 3)))
        time.sleep(3)