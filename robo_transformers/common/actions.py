

from gym.spaces import Box, Dict, Discrete
from robo_transformers.common.samples import Pose, PlanarDirection
from robo_transformers.interface import Sample, ControlAction, Control
from typing import SupportsFloat, Union, Sequence
from dataclasses import dataclass, field
import numpy as np
from beartype import beartype

@beartype
@dataclass
class PoseControl(Pose, ControlAction):
    '''Action for a 6D space representing x, y, z, roll, pitch, and yaw.
    '''
    def space(self) -> Dict:
        space = dict(Pose.space(self))
        space.update(ControlAction.space(self))
        return Dict(space)

@beartype
@dataclass
class PlanarDirectionControl(PlanarDirection, ControlAction):
    '''Action for a 2D+1 space representing x, y, and yaw.
    '''
    def space(self) -> Dict:
        space = dict(PlanarDirection.space(self))
        space.update(ControlAction.space(self))
        return Dict(space)

@beartype
@dataclass
class GraspControl(ControlAction):
    '''Action for a 1D space representing grip.
    '''
    grasp: SupportsFloat = 0
    grasp_bounds: Union[Sequence[SupportsFloat], np.ndarray] = field(default_factory=lambda: [-1, 1])

    def space(self) -> Dict:
        space = dict(ControlAction.space(self))
        space.update({
            'grasp': Box(low=self.grasp_bounds[0], high=self.grasp_bounds[1], shape=(), dtype=float),
        })
        return Dict(space)



@beartype
@dataclass
class GripperControl(ControlAction):
    '''Action for a 7D space representing x, y, z, roll, pitch, yaw, and grip of the gripper.
    '''
    pose: PoseControl = field(default_factory=PoseControl)
    grasp: GraspControl = field(default_factory=GraspControl)

    def __post_init__(self):
        '''Post initialization.
        '''
        if self.control_type != Control.UNSPECIFIED:
            assert self.pose.control_type in (self.control_type, Control.UNSPECIFIED), "Pose control type must match the gripper control type."
            assert self.grasp.control_type in (self.control_type, Control.UNSPECIFIED), "Grasp control type must match the gripper control type."

    def space(self):
        space = dict(ControlAction.space(self))
        space.update({'pose': self.pose.space()})
        space.update({'grasp': self.grasp.space()})
        return Dict(space)

@beartype
@dataclass
class JointControl(ControlAction):
    '''Action for a joint value, typically an angle.
    '''
    value: SupportsFloat = 0
    bounds: Union[Sequence[SupportsFloat], np.ndarray] =  field(default_factory=lambda: [-3.14, 3.14])

    def space(self) -> Dict:
        space = dict(ControlAction.space(self))
        space.update({
            'value': Box(low=self.bounds[0], high=self.bounds[1], shape=(), dtype=float),
        })
        return Dict(space)


@beartype
@dataclass
class FullJointControl(ControlAction):
    '''Action for joint control.
    '''
    joints: Union[Sequence[JointControl], np.ndarray] = field(default_factory=lambda: [JointControl()])
    names: Union[Sequence[str], np.ndarray, None] = None

    def __post_init__(self):
        '''Post initialization.
        '''
        if self.control_type != Control.UNSPECIFIED:
            for joint in self.joints:
                assert joint.control_type in (self.control_type, Control.UNSPECIFIED), "Joint control type must match the gripper control type."
                
    def space(self):
        space = ControlAction.space(self)
        if self.names is None:
            self.names = [f'joint_{i}' for i in range(len(self.joints))]

        for name, joint in zip(self.joints, self.names):
            space.update({name: joint.space()})
        return Dict(space)

@beartype
@dataclass
class EmbodiedControl(ControlAction):
    '''Action for arbitrary control of multiple actions.
    '''
    actions: Union[Sequence[ControlAction | Sample], np.ndarray] = field(default_factory=lambda: [ControlAction()])
    names: Union[Sequence[str], None] = None

    def __post_init__(self):
        '''Post initialization.
        '''
        if self.control_type != Control.UNSPECIFIED:
            for action in self.actions:
                assert action.control_type in (self.control_type, Control.UNSPECIFIED), "Action control type must match the gripper control type."

    def space(self):
        space = super(ControlAction,self).space()
        if self.names is None:
            self.names = [f'action_{i}' for i in range(len(self.actions))]

        for name, action in zip(self.actions, self.names):
            space.update({name: action.space()})
        return Dict(space)

@dataclass
class GripperBaseControl(ControlAction):
    '''Control for a gripper and optional right gripper and base. Defaults to relative control except for the gripper which defaults to absolute control with bounds [0,1].
    '''
    base: PlanarDirectionControl = field(default_factory= lambda: PlanarDirectionControl(control_type=Control.RELATIVE))
    left_gripper: GripperControl = field(default_factory= lambda: GripperControl(grasp=GraspControl(control_type=Control.ABSOLUTE, grasp_bounds=[0,1])))
    finish: bool = False

    right_gripper: GripperControl | None = None
    camera_tilt: JointControl | None = None
    camera_pan: JointControl | None = None

    def __post_init__(self):
        '''Post initialization.
        '''
        if self.control_type != Control.UNSPECIFIED:
            assert self.base.control_type in (self.control_type, Control.UNSPECIFIED), "Base control type must match the gripper control type."
            assert self.left_gripper.control_type in (self.control_type, Control.UNSPECIFIED), "Left gripper control type must match the gripper control type."
            if self.right_gripper is not None:
                assert self.right_gripper.control_type in (self.control_type, Control.UNSPECIFIED), "Right gripper control type must match the gripper control type."
        
    def space(self):
        space = dict(ControlAction.space(self))
        if self.right_gripper is not None:
            space.update({
                'right_gripper': self.right_gripper.space()
            })
        if self.camera_tilt is not None:
            space.update({
                'camera_tilt': self.camera_tilt.space()
            })
        if self.camera_pan is not None:
            space.update({
                'camera_pan': self.camera_pan.space()
            })
        space.update({
            'base': self.base.space(),
            'left_gripper': self.left_gripper.space(),
            'finish': Discrete(2)
        })
        return Dict(space)



