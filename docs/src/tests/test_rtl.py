from robo_transformers.models.rt1.inference import run_demo, load_rt1

def test_rt1_policy_main():
    policy = load_rt1(model_key='rt1main')
    assert run_demo(policy=policy) is None

def test_rt1_policy_simreal():
    policy = load_rt1(model_key='rt1simreal')
    assert run_demo(policy=policy) is None

def test_rt1_policy_multirobot():
    policy = load_rt1(model_key='rt1multirobot')
    assert run_demo(policy=policy) is None

def test_rt1_policy_x():
    policy = load_rt1(model_key='rt1x')
    assert run_demo(policy=policy) is None

if __name__ == '__main__':
    test_rt1_policy_main()
    test_rt1_policy_simreal()
    test_rt1_policy_multirobot()
    test_rt1_policy_x()
