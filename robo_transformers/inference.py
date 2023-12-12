from typing import Optional
import PIL


class Observation(object):
    pass
class State(object):
    pass
class Supervision(object):
    pass
class Config(object):
    pass


class Action(object):
    pass
class Actor(object):
    def act(self, observation: Observation, state: State, config: Config, supervision: Optional[Supervision] = None) -> Action:
        pass

class Src(object):
    '''The source data type from a sensor or web interface. A camera image, a point cloud, etc.'''
    pass

class InternalState(object):
    ''' The internal state of the agent. This is produced internally in contrast to Observation.'''
    pass

# This is different than the Observer in tf-agents. That observer simply records from the environment.
# This Observer is responsible from taking in data from the environment and converting it into an Observation.
# Essentially it is a tf-agents observer + a post-processor.

# In the case of RT1, the Observer would take in a camera image and a natural language instruction and convert it into an Observation.
class Observer(object):
    def observe(self, srcs: list[Src]) -> Observation:
        pass

class Rt1Observer(Observer):
    def observe(self, srcs: list[Src(PIL.Image), Src(str)]) -> Observation:
        pass


def inference(
    model: any,
    internal_state: dict,
    observation: dict,
    supervision: dict,
    config: dict,
) -> dict:
    """Infer action from observation.

    Args:
        cgn (CGN): ContactGraspNet model
        pcd (np.ndarray): point cloud
        threshold (float, optional): Success threshol. Defaults to 0.5.
        visualize (bool, optional): Whether or not to visualize output. Defaults to False.
        max_grasps (int, optional): Maximum grasps. Zero means unlimited. Defaults to 0.
        obj_mask (np.ndarray, optional): Object mask. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The grasps, confidence and indices of the points used for inference.
    """
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