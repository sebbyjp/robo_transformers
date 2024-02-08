from robo_transformers.inference_server import InferenceServer
from importlib.resources import files
from PIL import Image
import numpy as np
from einops import rearrange
demo_img = files("robo_transformers").joinpath("demo_imgs/gripper_almost_grasp.png")

def test_rt1pose():
    inference = InferenceServer('rt1/rt1pose', weights_path=None)
    img = np.array(Image.open(demo_img))
    action0 = inference(instruction="test", image=img, save=True)
    assert action0 is not None and action0 != {}
    assert len(action0.get('base_displacement_vector')) == 2
    assert len(action0.get('base_displacement_vertical_rotation')) == 1
    assert len(action0.get('gripper_closedness_action')) == 1
    assert len(action0.get('rotation_delta')) == 3
    assert len(action0.get('terminate_episode')) == 3
    assert len(action0.get('world_vector')) == 3

if __name__ == '__main__':
    test_rt1pose()
