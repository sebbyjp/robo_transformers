import tensorflow as tf
import numpy as np
import PIL.Image as Image
import tensorflow_hub as hub
from tf_agents.policies.py_tf_eager_policy import (
    SavedModelPyTFEagerPolicy as LoadedPolicy,
)
from tf_agents.trajectories import time_step as ts, policy_step as ps
from tf_agents import specs
from tf_agents.typing import types
from importlib.resources import files
from absl import logging, flags, app
import os
import gdown
import sys
from pprint import pprint
from typing import Optional

REGISTRY = {
    "rt1main": "https://drive.google.com/drive/folders/1QG99Pidaw6L9XYv1qSmuip_qga9FRcEC?usp=drive_link",
    "rt1simreal": "https://drive.google.com/drive/folders/1_nudHVmGuGUpGcrLlswg9O-aWy27Cjg0?usp=drive_link",
    "rt1multirobot": "https://drive.google.com/drive/folders/1EWjKSnfvD-ANPTLxugpCVP5zU6ADy8km?usp=drive_link",
    "rt1x": "https://drive.google.com/drive/folders/1LjTizUsqM88-5uHAIczTrObB3_z4OlgE?usp=drive_link",
    # 'xgresearch':
    #     "https://drive.google.com/drive/folders/185nP-a8z-1Pm6Zc3yU2qZ01hoszyYx51?usp=drive_link"
}

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "instruction", "pick up the block", "The instruction to run inference on."
)
flags.DEFINE_string(
    "model_key",
    "rt1simreal",
    "Which model to load. Must be one of: " + str(REGISTRY.keys()),
)
flags.DEFINE_string(
    "checkpoint_path", None, "Custom checkpoint path. This overrides the model key."
)

flags.DEFINE_boolean("show", False, "Whether or not to show the demo images.")

WIDTH = 320
HEIGHT = 256


class LazyTFModule:
    """Lazy loads a tensorflow module."""

    def __init__(self, url: str):
        self.url = url
        self.module = None

    def __getattr__(self, name: str):
        if self.module is None:
            self.module = hub.load(self.url)
        return getattr(self.module, name)

    def __call__(self, *args, **kwargs):
        if self.module is None:
            self.module = hub.load(self.url)
        return self.module(*args, **kwargs)


TEXT_ENCODER = LazyTFModule(
    "https://tfhub.dev/google/universal-sentence-encoder-large/5"
)


def download_checkpoint(key: str, output: str = None):
    if key not in REGISTRY.keys():
        logging.fatal("Invalid model key. Must be one of: ", REGISTRY.keys())

    if output is None:
        downloads_folder = os.path.join(os.getcwd(), "checkpoints/rt1/")
    else:
        downloads_folder = output

    output = os.path.join(downloads_folder, key)
    if not os.path.exists(output):
        logging.info("Downloading new model: ", key)
        gdown.download_folder(
            REGISTRY[key], output=downloads_folder, quiet=True, use_cookies=False
        )
    return output


def load_rt1(
    model_key: str = "rt1simreal",
    checkpoint_path: str = None,
    load_specs_from_pbtxt: bool = True,
    use_tf_function: bool = True,
    batch_time_steps: bool = False,
    downloads_folder: bool = None,
) -> LoadedPolicy:
    """Loads a trained RT-1 model from a checkpoint.

    Args:
        model_key (str, optional):  Model to load.
        checkpoint_path (str, optional): Custom checkpoint path.
        load_specs_from_pbtxt (bool, optional): Load from the .pb file in the checkpoint dir. Defaults to True.
        use_tf_function (bool, optional): Wraps function with optimized tf.Function to speed up inference. Defaults to True.
        batch_time_steps (bool, optional): Whether to automatically add a batch dimension during inference. Defaults to False.

    Returns:
        tf_agents.polices.tf_policy.TFPolicy: A tf_agents policy object.
    """
    # Suppress warnings from gdown and tensorflow.
    log_level = logging.get_verbosity()
    if log_level < 2:
        logging.set_verbosity(logging.ERROR)

    if checkpoint_path is None:
        checkpoint_path = download_checkpoint(model_key, downloads_folder)

    print("Loading RT-1 from checkpoint: {}...".format(checkpoint_path))

    policy: LoadedPolicy = LoadedPolicy(
        model_path=checkpoint_path,
        load_specs_from_pbtxt=load_specs_from_pbtxt,
        use_tf_function=use_tf_function,
        batch_time_steps=batch_time_steps,
    )
    print("RT-1 loaded.")

    logging.set_verbosity(log_level)
    return policy


def embed_text(input: list[str] | str, batch_size: int = 1) -> tf.Tensor:
    """Embeds a string using the Universal Sentence Encoder. Copies the string
        to fill the batch dimension.

    Args:
        input (str): The string to embed.
        batch_size (int, optional): . Defaults to 1.

    Returns:
        tf.Tensor: A tensor of shape (batch_size, 512).
    """
    if isinstance(input, str):
        input = input.lstrip(' ').rstrip(' ')
        input = np.tile(np.array(input), (batch_size,))
    embedded = TEXT_ENCODER(input).numpy()[0]
    return tf.reshape(
        tf.convert_to_tensor(embedded, dtype=tf.float32), (batch_size, 512)
    )


def format_images(images: np.ndarray | list[np.ndarray]) -> tf.Tensor:
    """Formats a list of images to the correct shape and type.

    Args:
        images (np.ndarray | list[np.ndarray]): Must have at most 4 dimensions.

    Returns:
        tf.Tensor: A tensor of shape (batch_size, HEIGHT, WIDTH, 3).
    """
    if isinstance(images, np.ndarray) and len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)

    out = []
    for image in images:
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.image.resize_with_pad(
            image, target_width=WIDTH, target_height=HEIGHT
        )
        image = tf.reshape(image, (1, HEIGHT, WIDTH, 3))
        image = tf.cast(image, dtype=tf.uint8)
        out.append(image)
    return tf.concat(out, 0)


def inference(
    instructions: list[str] | str,
    imgs: list[np.ndarray] | np.ndarray,
    step: int,
    reward: Optional[list[float] | float] = None,
    policy: Optional[LoadedPolicy] = None,
    policy_state=Optional[types.NestedArray],
    terminate=False,
) -> tuple[ps.ActionType, types.NestedSpecTensorOrArray, types.NestedSpecTensorOrArray]:
    """Runs inference on a list of images and instructions.

    Args:
        instructions (list[str]): A list of instructions. E.g. ["pick up the block"]
        imgs (list[np.ndarray]): A list of images with shape[(HEIGHT, WIDTH, 3)]
        step (int): The current time step.
        reward (list[float], optional): Defaults to None.
        policy (tf_agents.policies.tf_policy.TFPolicy, optional): Defaults to None.
        state (, optional). The internal network state. See 'policy state' in the "Data Types" section
            of README.md. Defaults to None.
        terminate (bool, optional): Whether or not to terminate the episode. Defaults to False.

    Returns:
        tuple[Action, State, Info]: The action, state, and info from the policy Again see the
         "Data Types" section of README.md.
    """
    if policy is None:
        policy = load_rt1()

    # Calculate batch size from instructions shape.
    if isinstance(instructions, str):
        batch_size = 1
        if reward is not None:
            reward = reward * np.ones((batch_size,), dtype=np.float32)
    else:
        batch_size = len(instructions)

    # Create the observation. RT-1 only reads the 'image' and 'natural_language_embedding' keys
    # so everything else can be zero.
    observation = specs.zero_spec_nest(
        specs.from_spec(policy.time_step_spec.observation), outer_dims=(batch_size,)
    )
    
    if reward is None:
        reward = np.zeros((batch_size,), dtype=np.float32)

    if policy_state is None:
         # Run dummy inference to get the initial state.
        policy_state = policy.get_initial_state(batch_size)
        time_step = ts.transition(observation, np.zeros((batch_size,), dtype=np.float32))
        _, policy_state, _ = policy.action(time_step, policy_state)

    observation["image"] = format_images(imgs)
    observation['natural_language_embedding'] = embed_text(
        instructions, batch_size)

    if step == 0:
        time_step = ts.restart(observation, batch_size)
    elif terminate:
        time_step = ts.termination(observation, reward)
    else:
        time_step = ts.transition(observation, reward)

    action, next_state, info = policy.action(time_step, policy_state)

    if logging.level_debug():
        writer = tf.summary.create_file_writer("./runs")
        with writer.as_default():
            for i in range(3):
                tf.summary.scalar(
                    "world_vector{}".format(i), action["world_vector"][0, i], step=step
                )
                tf.summary.scalar(
                    "rotation_delta{}".format(i),
                    action["rotation_delta"][0, i],
                    step=step,
                )
            tf.summary.scalar(
                "gripper_closedness_action{}".format(i),
                action["gripper_closedness_action"][0, 0],
                step=step,
            )
            writer.flush()

    return action, next_state, info


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
        if FLAGS.show and output is not None:
            img.save(os.path.join(output, fn.name))
        img = np.array(img.convert("RGB"))
        images.append(img)
    logging.set_verbosity(log_level)
    return images


def run_demo(policy: LoadedPolicy = None):
    if policy is None:
        policy = load_rt1(FLAGS.model_key, FLAGS.checkpoint_path)

    # Pass in an instruction through the --instructions flag.
    # The rewards will not affect the inference at test time.
    images = get_demo_images(output=os.getcwd())
    rewards = [0, 0, 0]
    state = None
    for step in range(3):
        action, state, _ = inference(
            FLAGS.instruction,
            images[step],
            step,
            rewards[step],
            policy,
            policy_state=state,
            terminate=(step == 2),
        )
        pprint(action)
        print(" ")


def main(_):
    if logging.level_debug():
        tf.debugging.experimental.enable_dump_debug_info(
            "./runs", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1
        )

    # Run three time steps of inference using the demo images.
    # Pass in an instruction via the command line.
    run_demo()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            """
        RT-1 Inference Demo
        -------------------
        This demo runs inference on a pretrained RT-1 model. It uses a pretrained
        Universal Sentence Encoder to embed the instruction and a pretrained
        ViT to extract features from the image.

        The demo will run inference on three images from the demo_imgs directory
        and print the output to the console.

        You can also pass in a custom instruction via the --instruction flag.

        To run the demo, use the following command:
        python3 -m robo_transformers.rt1.inference --model_key=rt1simreal --instruction="pick block"
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
