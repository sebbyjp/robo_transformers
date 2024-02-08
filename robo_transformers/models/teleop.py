from robo_transformers.abstract.agent import Agent
from robo_transformers.models.octo.action import OctoAction
from typing import Optional
import os
import cv2
import numpy as np
import numpy.typing as npt
from beartype import beartype
import math
from robo_transformers.recorder import Recorder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@beartype
class TeleOpAgent(Agent):
    def __init__(
        self,
        xyz_step: float = 0.01,
        rpy_step: float = math.pi / 8,
        record_dir: str = "episodes",
    ) -> None:
        """Agent for octo model.

        Args:
            xyz_step (float, optional): Step size for xyz. Defaults to 0.02.
            rpy_step (float, optional): Step size for rpy. Defaults to PI/8.
        """
        self.xyz_step = xyz_step
        self.rpy_step = rpy_step
        self.recorder = Recorder("episode1", data_dir=record_dir)
        # self.replayer = Replayer("episode1", data_dir="episodes")
    
    def process_input(self):
        value = input(
            """
            Please enter one or more values for the action. The following correspond to a pose delta in the world frame:\n
            w = forward x (+x) \n
            s = backward x (-x) \n
            a = left y (+y)\n
            d = right y (-y)\n
            x = up z (+z) \n
            z = down z (-z) \n
            q = close gripper\n
            e = open gripper\n
            shift+d = roll right (-r) \n
            shift+a = roll left (+r)\n
            shift+w = pitch up (-p)\n
            shift+s = pitch down (+p)\n
            shift+z = yaw left (+y)\n
            shift+x = yaw right (-y)\n
            c = complete episode successfully\n
            f = finish episode with failure\n

            Type each command as many times as needed. Each will correspond to a single step (default xyz: 0.01, default rpy: PI/8.\n
            Press enter to continue.\n
            """
        )

        grasp = 0.
        if "q" in value:
            grasp = -1.
        elif "e" in value:
            grasp = 1.
        else:
            grasp = 0.

        reward = float("c" in value)
        done = float(reward or "f" in value)

        xyz = self.xyz_step * np.array(
            [
                value.count("w") - value.count("s"),
                value.count("a") - value.count("d"),
                value.count("x") - value.count("z"),
            ]
        )
        rpy = self.rpy_step * np.array(
            [
                value.count("D") - value.count("A"),
                value.count("S") - value.count("W"),
                value.count("Z") - value.count("X"),
            ]
        )
        action = np.concatenate([xyz, rpy, [grasp]])
        return action, reward, done

    def act(
        self,
        instruction: str,
        image: npt.ArrayLike,
        image_wrist: Optional[npt.ArrayLike] = None,
    ) -> OctoAction:
        # Create observation of past `window_size` number of observations
        image = cv2.resize(np.array(image, dtype=np.uint8), (224, 224))
        if image_wrist is not None:
            image_wrist = cv2.resize(np.array(image_wrist, dtype=np.uint8), (128, 128))

        action, reward, done = self.process_input()
        # if sum(action) == 0:
        #     print('replaying')
        #     action = next(self.replayer)
        # else:
        #     next(self.replayer)

        print("action: ", action)
        self.recorder.record(
            observation={
                "image_head": image,
                "image_wrist_left": image_wrist,
                "language_instruction": instruction.encode(),
            },
            action={
                "left_hand": {
                    "x": action[0],
                    "y": action[1],
                    "z": action[2],
                    "roll": action[3],
                    "pitch": action[4],
                    "yaw": action[5],
                    "grasp": action[6],
                    "encoding": 0
                }
            },
            reward=reward,
            done=done,
        )
        if done:
            actions = np.array(
                [
                    np.array(
                        [
                            self.recorder.file["action/left_hand/x"][i],
                            self.recorder.file["action/left_hand/y"][i],
                            self.recorder.file["action/left_hand/z"][i],
                            self.recorder.file["action/left_hand/roll"][i],
                            self.recorder.file["action/left_hand/pitch"][i],
                            self.recorder.file["action/left_hand/yaw"][i],
                            self.recorder.file["action/left_hand/grasp"][i],
                        ]
                    )
                    for i in range(self.recorder.file.attrs["size"])])
 
            print(f"mean: {np.mean(actions, axis=0)}")
            print(f"std: {np.std(actions, axis=0)}")
            self.recorder.close()
        
        return OctoAction(*action)
