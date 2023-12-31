from tf_agents.policies.py_policy import PyPolicy
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import policy_step as ps
from robo_transformers.rt1.rt1_inference import load_rt1, inference as rt1_inference
from PIL import Image
import numpy as np
from typing import Optional
from pprint import pprint
import tensorflow as tf
from absl import logging

def rescale_action_with_bound(
    actions: tf.Tensor,
    low: float,
    high: float,
    safety_margin: float = 0,
    post_scaling_max: float = 1.0,
    post_scaling_min: float = -1.0,
) -> tf.Tensor:
  """Formula taken from https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range."""
#   resc_actions = (actions - low) / (high - low) * (
#       post_scaling_max - post_scaling_min
#   ) + post_scaling_min
  return tf.clip_by_value(
      actions,
     low,
   high,
  )

def rescale_action(action):
  """Rescales action."""

  action['world_vector'] = rescale_action_with_bound(
      action['world_vector'],
      low=-0.05,
      high=0.05,
      safety_margin=0.01,
    #   post_scaling_max=1.75,
    #   post_scaling_min=-1.75,
  )
  action['rotation_delta'] = rescale_action_with_bound(
      action['rotation_delta'],
      low=-0.25,
      high=0.25,
      safety_margin=0.01,
    #   post_scaling_max=1.4,
    #   post_scaling_min=-1.4,
  )

class InferenceServer:

    def __init__(self,
                 model_key: str = "rt1main",
                 model: Optional[PyPolicy | TFPolicy] = None,
                 pass_through: bool = False):
        self.pass_through = pass_through
        if pass_through:
            return
    
        self.policy_state = None
        self.step = 0

        self.model = model
        if self.model is None:
            self.model = load_rt1(model_key=model_key)

 

    def __call__(self,
                 instructions: list[str] | str,
                 imgs: list[np.ndarray] | np.ndarray,
                 reward: list[float] | float = None,
                 terminate: bool = False,
                 save: bool = False,
                 return_policy_state: bool = False) -> ps.ActionType:
        imgs = np.array(imgs, dtype=np.uint8)
        # imgs = np.array(
        #         Image.fromarray(imgs).resize((WIDTH, HEIGHT)).convert('RGB'))
        # if (len(imgs.shape) not in [3, 4] or
        #     (len(imgs.shape) == 3 and
        #      (imgs.shape[0] != WIDTH or imgs.shape[1] != HEIGHT or
        #       imgs.shape[2] != 3)) or len(imgs.shape) == 4 and
        #     (imgs.shape[1] != WIDTH or imgs.shape[2] != HEIGHT or
        #      imgs.shape[3] != 3)):

        #     imgs = np.array(
        #         Image.fromarray(imgs).resize((WIDTH, HEIGHT)).convert('RGB'))
        if save:
            Image.fromarray(imgs).save("rt1_saved_img.png")

        if self.pass_through:
            return {
                'base_displacement_vector':
                    np.array([0.0, 0.0], dtype=np.float32),
                'base_displacement_vertical_rotation':
                    np.array([0.0], dtype=np.float32),
                'gripper_closedness_action':
                    np.array([0.0], dtype=np.float32),
                'rotation_delta':
                    np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'terminate_episode':
                    np.array([0, 0, 0], dtype=np.int32),
                'world_vector':
                    np.array([0.02, 0.00, -0.02], dtype=np.float32),
            }
        try:
            action, state, _ = rt1_inference(instructions, imgs, self.step, reward,
                                         self.model, self.policy_state,
                                         terminate)
            self.policy_state = state
            self.step += 1
            if logging.level_debug():
                print('before rescaling action')
                pprint(action)
            # rescale_action(action)
            action = {
                'base_displacement_vector':
                    np.array(action['base_displacement_vector'], dtype=np.float32),
                'base_displacement_vertical_rotation':
                    np.array(action['base_displacement_vertical_rotation'], dtype=np.float32),
                'gripper_closedness_action':
                    np.array(action['gripper_closedness_action'], dtype=np.float32),
                'rotation_delta':
                    np.array(action['rotation_delta'], dtype=np.float32),
                'terminate_episode':
                    np.array(action['terminate_episode'], dtype=np.int32),
                'world_vector':
                    np.array(action['world_vector'], dtype=np.float32),
            }
            pprint(action)

            if return_policy_state:
                return action, self.policy_state
            else:
                return action
        except Exception as e:
            import traceback
            traceback.print_tb(e.__traceback__)
      
  



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
