from tf_agents.policies.py_policy import PyPolicy
from tf_agents.policies.tf_policy import TFPolicy
from tf_agents.trajectories import policy_step as ps
from robo_transformers.rt1.rt1_inference import load_rt1, inference as rt1_inference
import numpy as np


class InferenceServer:

    def __init__(self,
                 model: PyPolicy | TFPolicy = None,
                 verbose: bool = False):
        self.model = model
        if self.model is None:
            self.model = load_rt1()

        self.policy_state = None
        self.verbose = verbose
        self.step = 0

    def __call__(self,
                 instructions: list[str] | str,
                 imgs: list[np.ndarray] | np.ndarray,
                 reward: list[float] | float = None,
                 terminate: bool = False) -> ps.ActionType:
        action, state, _ = rt1_inference(instructions, imgs, self.step, reward,
                                  self.model, self.policy_state, terminate,
                                  self.verbose)
        self.policy_state = state
        self.step += 1
        return action



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
