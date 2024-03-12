from PIL import Image
import numpy as np
from typing import Optional, TypeVar
from pprint import pprint
from absl import logging
from beartype import beartype
from dataclasses import asdict, fields
from numpy.typing import ArrayLike
from robo_transformers.registry import REGISTRY
from robo_transformers.abstract.action import  Action
from robo_transformers.abstract.agent import Agent

ActionT = TypeVar('ActionT', bound=Action)

@beartype
class InvertingDummyAction:
    '''Returns Dummy Action that inverts its action every call.
    '''
    def __init__(self, action: ActionT):
        self.action: ActionT = action

    def invert_(self):
        for f in fields(self.action):
            setattr(self.action, f.name, -getattr(self.action, f.name))

    def __post_init__(self):
        # First call will invert again.
        self.invert_()

    def make(self) -> dict:
        self.invert_()
        return asdict(self.action)
        

@beartype
class InferenceServer:

    def __init__(self,
                 model_type: str = "rt1",
                 weights_key: str = "rt1main",
                 dummy: bool = False,
                 agent: Optional[Agent] = None,
                 **kwargs
                 ):
        '''Initializes the inference server.


        Args:
            model_type (str, optional): Defaults to "rt1".
            weights_key (str, optional): Defaults to "rt1main".
            dummy (bool, optional): If true, a dummy action will be returned that inverts every call. Defaults to False.
            agent (VLA, optional): Custom agent that implements VLA interface. Defaults to None.
            **kwargs: kwargs for custom agent initialization.

        '''

        self.dummy: bool = dummy

        if dummy:
            self.action = InvertingDummyAction(REGISTRY[model_type]['action']())
            return
        elif agent is not None:
            self.agent: Agent = agent
        else:
            self.agent: Agent = REGISTRY[model_type]['agent'](weights_key)
            self.action = REGISTRY[model_type]['action']()

    def __call__(self,
                 save: bool = False,
                 **kwargs
                 ) -> dict:
        '''Runs inference on a Vision Language Action model.


        Args:
            save (bool, optional): Whether or not to save the observed image. Defaults to False.
            *args: args for the agent.
            **kwargs: kwargs for the agent.

        Returns:
            dict: See RT1Action for details.
        '''
        image: ArrayLike = kwargs.get('image')
        if image is not None and save:
            Image.fromarray(np.array(image, dtype=np.uint8)).save("rt1_saved_image.png")

        if not self.dummy:
            try:
                self.action = self.agent.act(**kwargs)

                if logging.get_verbosity() > logging.DEBUG and kwargs.get('instruction') is not None:
                    intruction = kwargs['instruction']
                    print(f'instruction: {intruction}')
                    pprint(self.action)

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
