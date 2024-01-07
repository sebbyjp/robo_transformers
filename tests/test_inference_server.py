from robo_transformers.inference_server import InferenceServer
from importlib.resources import files
from PIL import Image
import numpy as np

demo_img = files("robo_transformers").joinpath("demo_imgs/gripper_almost_grasp.png")

def test_dummy():
    inference = InferenceServer(dummy=True)
    img = np.array(Image.open(demo_img))
    action0 = inference(instructions="test", images=img, save=True)
    assert action0 is not None
    assert len(action0.get('base_displacement_vector')) == 2
    assert len(action0.get('base_displacement_vertical_rotation')) == 1
    assert len(action0.get('gripper_closedness_action')) == 1
    assert len(action0.get('rotation_delta')) == 3
    assert len(action0.get('terminate_episode')) == 3
    assert len(action0.get('world_vector')) == 3

    action1 = inference(instructions="test", images=img, save=True)
    for k, v in action1.items():
        assert np.array_equal(v ,-action0[k])
    
    action2 = inference(instructions="test", images=img, save=True)
    for k, v in action2.items():
        assert np.array_equal(v ,-action1[k])

if __name__ == '__main__':
    test_dummy()
