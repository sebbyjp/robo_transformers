from robo_transformers.rt1_inference import run_on_test_imgs

def test_rt_1_policy():
    assert run_on_test_imgs() is None

if __name__ == '__main__':
    test_rt_1_policy()
