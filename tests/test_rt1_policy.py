from robo_transformers.rt1.rt1_inference import run_on_demo_imgs, load_rt1
from importlib.resources import files

checkpoints_dir = files("robotics_transformer").joinpath("trained_checkpoints/")
def test_rt_1_policy_main():
    policy = load_rt1(checkpoints_dir.joinpath('rt1main'))
    assert run_on_demo_imgs(rt1_policy=policy) is None

def test_rt_1_policy_1_simreal():
    policy = load_rt1(checkpoints_dir.joinpath('rt1simreal'))
    assert run_on_demo_imgs(rt1_policy=policy) is None

def test_rt_1_policy_1_multirobot():
    policy = load_rt1(checkpoints_dir.joinpath('rt1multirobot'))
    assert run_on_demo_imgs(rt1_policy=policy) is None

if __name__ == '__main__':
    test_rt_1_policy_main()
    test_rt_1_policy_1_simreal()
    test_rt_1_policy_1_multirobot()
