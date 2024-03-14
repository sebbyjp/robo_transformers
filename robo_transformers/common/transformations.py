''' Transformations between spaces. '''

# from robo_transformers.common.samples import Pose, PlanarDirection
# from robo_transformers.common.actions import PoseControl, PlanarDirectionControl, JointControl, GripperControl

# def linear_to_loco(linear: float, rate: float, time: float, step_size: float=0.04) -> float:
#   '''linear -> loco
#   '''
#   return linear / rate / time / step_size
# def rt1_to_motion_control(rt1: GripperControl) -> dict:

#   '''rt1 -> motion control
#   '''
  
#   return {
#     'base_x_cmd': rt1.pose.xyz[0]
#     'base_theta_cmd': rt1.get('base_theta_cmd'),
#     'pan_cmd': rt1.get('pan_cmd'),
#     'tilt_cmd': rt1.get('tilt_cmd'),
#     'ee_x_cmd': rt1.get('ee_x_cmd'),
#     'ee_y_cmd': rt1.get('ee_y_cmd'),
#     'ee_z_cmd': rt1.get('ee_z_cmd'),
#     'ee_roll_cmd': rt1.get('ee_roll_cmd'),
#     'ee_pitch_cmd': rt1.get('ee_pitch_cmd'),
#     'waist_cmd': rt1.get('waist_cmd'),
#     'gripper_cmd': rt1.get('gripper_cmd'),
#   }
