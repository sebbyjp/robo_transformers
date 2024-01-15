import tensorflow as tf
import numpy as np
import PIL.Image as Image
from robo_transformers.registry import REGISTRY
from importlib.resources import files
from absl import logging, flags, app
import os
import sys
from pprint import pprint

from robo_transformers.abstract.agent import Agent
from robo_transformers.abstract.action import Action


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "demo_instruction", "pick up the block", "The instruction to run inference on."
)
flags.DEFINE_string(
    "model_type",
    "octo",
    "Which model to load. Must be one of: " + str(REGISTRY.keys()),
)


flags.DEFINE_boolean("show_images", False, "Whether or not to show the demo images.")

def get_demo_images(output=None) -> np.ndarray:
    """Loads a demo video from the directory.

    Returns:
        list[tf.Tensor]: A list of tensors of shape (batch_size, HEIGHT, WIDTH, 3).
    """
    # Suppress noisy PIL warnings.
    log_level = logging.get_verbosity()
    if logging.get_verbosity() < 2:
        logging.set_verbosity(logging.ERROR)

    filenames = [
        files("robo_transformers").joinpath("demo_imgs/gripper_far_from_grasp.png"),
        files("robo_transformers").joinpath("demo_imgs/gripper_mid_to_grasp.png"),
        files("robo_transformers").joinpath("demo_imgs/gripper_almost_grasp.png"),
    ]

    images = []
    for fn in filenames:
        img = Image.open(fn)
        if FLAGS.show_images and output is not None:
            img.save(os.path.join(output, fn.name))
        img = np.array(img.convert("RGB"))
        images.append(img)
    logging.set_verbosity(log_level)
    return images


def run_demo(agent: Agent) -> list[Action]:

    # Pass in an instruction through the --demo_instruction flag.
    actions = []
    images = get_demo_images(output=os.getcwd())
    for step in range(3):
        action = agent.act(
            FLAGS.demo_instruction,
            images[step],
        )
        pprint(action)
        print(" ")
        actions.append(action)
    return actions


def main(_):
    if logging.level_debug():
        tf.debugging.experimental.enable_dump_debug_info(
            "./runs", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1
        )

    # Run three time steps of inference using the demo images.
    # Pass in an instruction via the command line.
    agent: Agent = REGISTRY[FLAGS.model_type]['agent']()
    run_demo(agent)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            """
        Octo Inference Demo
        -------------------
        This demo runs inference on a pretrained Octo model.

        The demo will run inference on three images from the demo_imgs directory
        and print the output to the console.

        You can also pass in a custom instruction via the --instruction flag.

        To run the demo, use the following command:
        python3 -m robo_transformers.demo --model_type=octo --instruction="pick block"
        """
        )
    app.run(main)
else:
    # TODO (speralta): Consider reading in flags from argv
    # CAREFUL: This will crash if you allow flags from argv that haven't been
    # defined.
    # For now, only apps that use app.run() can set flags.
    # Read in the program name
    FLAGS(sys.argv[0:1])

