from tf_agents.policies.py_policy import PyPolicy
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import policy_step as ps
from robo_transformers.rt1.rt1_inference import load_rt1, inference as rt1_inference
from PIL import Image
import numpy as np
from typing import Optional, TypedDict
from pprint import pprint
import tensorflow as tf
import torch
from absl import logging
from dataclasses import dataclass, field, asdict, fields
from collections.abc import Sequence
from nptyping import NDArray, Shape, Float, Int
from beartype import beartype

ArrayLike = NDArray | torch.Tensor | tf.Tensor | Sequence

class RT1ActionDictT(TypedDict):
    base_displacement_vector: ArrayLike
    base_displacement_vertical: ArrayLike
    gripper_closedness_action: ArrayLike
    rotation_delta: ArrayLike
    terminate_episode: ArrayLike
    world_vector: ArrayLike


@beartype
@dataclass
class RT1Action:
    base_displacement_vector: np.ndarray = field(default_factory= lambda: np.array([0.0, 0.0]))
    base_displacement_vertical_rotation:  np.ndarray   = field(default_factory= lambda : np.array([0.0]))
    gripper_closedness_action: np.ndarray  = field(default_factory= lambda : np.array([1.0]))
    rotation_delta:  np.ndarray   = field(default_factory= lambda : np.array([0.0, 0.0, 0.0]))
    terminate_episode: np.ndarray   = field(default_factory= lambda : np.array([0,0,0]))
    world_vector:  np.ndarray  = field(default_factory= lambda : np.array([0.0, 0.0, 0.02]))
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(**{k: np.squeeze(v.numpy(), axis=0) for k, v in d.items()})

    def make(self) -> RT1ActionDictT:
        return asdict(self)


@beartype
@dataclass
class InvertingDummyAction(RT1Action):
    '''Returns Dummy Action that inverts its action every call.
    '''
    def invert_(self):
        for f in fields(self):
            setattr(self, f.name, -getattr(self, f.name))

    def __post_init__(self):
        # First call will invert again.
        self.invert_()

    def make(self) -> RT1ActionDictT:
        self.invert_()
        return asdict(self)
        

@beartype
class InferenceServer:

    def __init__(self,
                 model_key: str = "rt1main",
                 model: Optional[PyPolicy | TFPolicy] = None,
                 dummy: bool = False):
        self.dummy: bool = dummy
        self.policy_state: Optional[dict] = None
        self.step: int = 0
        self.model: PyPolicy | TFPolicy = model
        self.action: RT1Action = RT1Action()

        if dummy:
            self.action = InvertingDummyAction()
            return

        if model is None:
            self.model = load_rt1(model_key=model_key)

    def __call__(self,
                 instructions: ArrayLike | str,
                 images: ArrayLike,
                 reward: Optional[ArrayLike | float] = None,
                 save: bool = False,
                 ) -> RT1ActionDictT:
        '''Runs inference on RT-1.


        Args:
            instructions (list[str] | str): Natural language instruction
            images (list[np.ndarray] | np.ndarray)
            reward (Optional[list[float]  |  float], optional): Defaults to None.
            save (bool, optional): _description_. Defaults to False.
            translation_bounds (tuple[float, float], optional). Defaults to (-0.05, 0.05).
            rotation_bounds (tuple[float, float], optional). Defaults to (-0.25, 0.25).

        Returns:
            dict: See RT1Action
        '''
        images = np.array(images, dtype=np.uint8)
        if save:
            Image.fromarray(images).save("rt1_saved_image.png")

        if isinstance(instructions, str):
            instructions = [instructions]
            images = [images]
        
        if not self.dummy:
            try:
                action, state, _ = rt1_inference(instructions, images, self.step,
                                                reward, self.model,
                                                self.policy_state)                          
                self.policy_state = state
                self.step += 1
                self.action = RT1Action.from_dict(action)
                

                if logging.get_verbosity() > logging.DEBUG:
                    print(f'instruction: {instructions}')
                    pprint(action)

            except Exception as e:
                import traceback
                traceback.print_tb(e.__traceback__)
                raise e

        return self.action.make()





# class Rt1Observer(Observer):
#     def observe(self, srcs: list[Src(PIL.Image), Src(str)]) -> Observation:
#         pass

# def inference(
#     model: any,
#     internal_state: dict,
#     observation: dict,
#     supervision: dict,
#     config: dict,
# ) -> dict:
#     """Infer action from observation.

#     Args:
#         cgn (CGN): ContactGraspNet model
#         pcd (np.ndarray): point cloud
#         threshold (float, optional): Success threshol. Defaults to 0.5.
#         visualize (bool, optional): Whether or not to visualize output. Defaults to False.
#         max_grasps (int, optional): Maximum grasps. Zero means unlimited. Defaults to 0.
#         obj_mask (np.ndarray, optional): Object mask. Defaults to None.

#     Returns:
#         tuple[np.ndarray, np.ndarray, np.ndarray]: The grasps, confidence and indices of the points used for inference.
#     """
# cgn.eval()
# pcd = torch.Tensor(pcd).to(dtype=torch.float32).to(cgn.device)
# if pcd.shape[0] > 20000:
#     downsample_idxs = np.array(random.sample(range(pcd.shape[0] - 1), 20000))
# else:
#     downsample_idxs = np.arange(pcd.shape[0])
# pcd = pcd[downsample_idxs, :]

# batch = torch.zeros(pcd.shape[0]).to(dtype=torch.int64).to(cgn.device)
# fps_idxs = farthest_point_sample(pcd, batch, 2048 / pcd.shape[0])

# if obj_mask is not None:
#     obj_mask = torch.Tensor(obj_mask[downsample_idxs])
#     obj_mask = obj_mask[fps_idxs]
# else:
#     obj_mask = torch.ones(fps_idxs.shape[0])
# points, pred_grasps, confidence, pred_widths, _, _ = cgn(
#     pcd[:, 3:],
#     pcd_poses=pcd[:, :3],
#     batch=batch,
#     idxs=fps_idxs,
#     gripper_depth=gripper_depth,
#     gripper_width=gripper_width,
# )

# sig = torch.nn.Sigmoid()
# confidence = sig(confidence)
# confidence = confidence.reshape(-1)
# pred_grasps = (
#     torch.flatten(pred_grasps, start_dim=0, end_dim=1).detach().cpu().numpy()
# )

# confidence = (
#     obj_mask.detach().cpu().numpy() * confidence.detach().cpu().numpy()
# ).reshape(-1)
# pred_widths = (
#     torch.flatten(pred_widths, start_dim=0, end_dim=1).detach().cpu().numpy()
# )
# points = torch.flatten(points, start_dim=0, end_dim=1).detach().cpu().numpy()

# success_mask = (confidence > threshold).nonzero()[0]
# if len(success_mask) == 0:
#     print("failed to find successful grasps")
#     return None, None, None

# success_grasps = pred_grasps[success_mask]
# success_confidence = confidence[success_mask]
# print("Found {} grasps".format(success_grasps.shape[0]))
# if max_grasps > 0 and success_grasps.shape[0] > max_grasps:
#     success_grasps = success_grasps[:max_grasps]
#     success_confidence = success_confidence[:max_grasps]
# if visualize:
#     visualize_grasps(
#         pcd.detach().cpu().numpy(),
#         success_grasps,
#         gripper_depth=gripper_depth,
#         gripper_width=gripper_width,
#     )
# return success_grasps, success_confidence, downsample_idxs
