from robo_transformers.abstract.agent import Agent
from tf_agents.policies.py_policy import PyPolicy
from tf_agents.policies.tf_policy import TFPolicy
from robo_transformers.models.rt1.inference import load_rt1, inference
from robo_transformers.models.rt1.action import RT1Action
from typing import Optional
import numpy.typing as npt
import numpy as np
from beartype import beartype


# Agent for RT1
@beartype
class RT1Agent(Agent):
    def __init__(self, weights_key: str) -> None:
        self.model: PyPolicy | TFPolicy = load_rt1(model_key=weights_key)
        self.policy_state: Optional[dict]   = None
        self.step_num: int = 0
    
    def act(self, instruction: str, image: npt.ArrayLike, reward: float = 0.0) -> RT1Action:
        image = np.array(image, dtype=np.uint8)
        action, next_state, _ = inference(instruction, image, self.step_num, reward, self.model, self.policy_state)
        self.step_num += 1
        self.policy_state = next_state

        return RT1Action.from_numpy_dict(action)

    