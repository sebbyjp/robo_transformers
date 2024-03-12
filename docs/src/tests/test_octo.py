from robo_transformers.demo import run_demo
from robo_transformers.models.octo.agent import OctoAgent
import numpy as np

def test_octo_agent():
    """
    Test OctoAgent
    """
    agent = OctoAgent()
    run_demo(agent)

def test_octo_agent_with_wrist():
    """
    Test OctoAgent with wrist
    """
    agent = OctoAgent()
    image = np.ones((224, 224, 3), dtype=np.uint8)
    image_wrist = np.ones((128, 128, 3), dtype=np.uint8)
    action = agent.act(
        "Do somethings",
        image,
        image_wrist,
    )
    assert action is not None and action.make() != {}





if __name__ == '__main__':
    actions = test_octo_agent()
    for action in actions:
        assert action is not None and action.make() != {}