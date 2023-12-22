from robo_transformers.inference_server import InferenceServer
from importlib.resources import files
from PIL import Image
import numpy as np
demo_img = files("robotics_transformer").joinpath("demo_imgs/gripper_almost_grasp.png")
def test_pass_through():
    inference = InferenceServer(pass_through=True)
    img = np.array(Image.open(demo_img))
    action = inference(instructions="test", imgs=img, save=True)
    assert action is not None
    assert len(action.get('base_displacement_vector')) == 2
    assert len(action.get('base_displacement_vertical_rotation')) == 1
    assert len(action.get('gripper_closedness_action')) == 1
    assert len(action.get('rotation_delta')) == 3
    assert len(action.get('terminate_episode')) == 3
    assert len(action.get('world_vector')) == 3

if __name__ == '__main__':
    test_pass_through()
