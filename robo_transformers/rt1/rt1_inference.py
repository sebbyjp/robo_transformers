import tensorflow as tf
from pprint import pprint
import numpy as np
import PIL.Image as Image
import tensorflow_hub as hub
from tf_agents.policies.py_tf_eager_policy import SavedModelPyTFEagerPolicy as LoadedPolicy
from tf_agents.trajectories import time_step as ts
from tf_agents import specs
import absl.logging
import os
import gdown
import argparse

WIDTH = 320
HEIGHT = 256
TEXT_ENCODER = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

REGISTRY = {
    'rt1main':
        "https://drive.google.com/drive/folders/1QG99Pidaw6L9XYv1qSmuip_qga9FRcEC?usp=drive_link",
    'rt1simreal':
        "https://drive.google.com/drive/folders/1_nudHVmGuGUpGcrLlswg9O-aWy27Cjg0?usp=drive_link",
    'rt1multirobot':
        "https://drive.google.com/drive/folders/1EWjKSnfvD-ANPTLxugpCVP5zU6ADy8km?usp=drive_link",
    'xlatest':
        "https://drive.google.com/drive/folders/1LjTizUsqM88-5uHAIczTrObB3_z4OlgE?usp=drive_link",
    'xgresearch':
        "https://drive.google.com/drive/folders/185nP-a8z-1Pm6Zc3yU2qZ01hoszyYx51?usp=drive_link"
}


def download_checkpoint(key: str, output: str = None):
    if key not in REGISTRY.keys():
        raise Exception('Invalid model key. Must be one of: ', REGISTRY.keys())

    if output is None:
        downloads_folder = os.path.join(os.getcwd(), 'checkpoints/rt1/')
    else:
        downloads_folder = output

    output = os.path.join(downloads_folder, key)
    if not os.path.exists(output):
        print('Downloading new model: ', key)
        gdown.download_folder(REGISTRY[key],
                              output=downloads_folder,
                              quiet=True,
                              use_cookies=False)
        #   quiet=True)
    return output


def load_rt1(model_key: str = 'rt1simreal',
             checkpoint_path: str = None,
             load_specs_from_pbtxt: bool = True,
             use_tf_function: bool = True,
             batch_time_steps: bool = False,
             verbose: bool = False,
             downloads_folder: bool = None) -> LoadedPolicy:
    '''Loads a trained RT-1 model from a checkpoint.

    Args:
        model_key (str, optional):  Model to load. 
        checkpoint_path (str, optional): Custom checkpoint path.
        load_specs_from_pbtxt (bool, optional): Load from the .pb file in the checkpoint dir. Defaults to True.
        use_tf_function (bool, optional): Wraps function with optimized tf.Function to speed up inference. Defaults to True.
        batch_time_steps (bool, optional): Whether to automatically add a batch dimension during inference. Defaults to False.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        tf_agents.polices.tf_policy.TFPolicy: A tf_agents policy object.
    '''
    if not verbose:
        absl.logging.set_verbosity(absl.logging.ERROR)
    if checkpoint_path is None:
        checkpoint_path = download_checkpoint(model_key, downloads_folder)

    print('Loading RT-1 from checkpoint: {}...'.format(checkpoint_path))

    policy: LoadedPolicy = LoadedPolicy(
        model_path=checkpoint_path,
        load_specs_from_pbtxt=load_specs_from_pbtxt,
        use_tf_function=use_tf_function,
        batch_time_steps=batch_time_steps)
    absl.logging.set_verbosity(absl.logging.WARN)
    print('RT-1 loaded.')
    return policy


def embed_text(input: list[str] | str, batch_size: int = 1) -> tf.Tensor:
    '''Embeds a string using the Universal Sentence Encoder.

    Args:
        input (str): The string to embed.
        batch_size (int, optional): . Defaults to 1.

    Returns:
        tf.Tensor: A tensor of shape (batch_size, 512).
    '''
    if isinstance(input, str):
        input = np.tile(np.array(input), (batch_size,))
    embedded = TEXT_ENCODER(input).numpy()[0]
    return tf.reshape(tf.convert_to_tensor(embedded, dtype=tf.float32),
                      (batch_size, 512))


def get_demo_imgs() -> tf.Tensor:
    '''Loads a demo video from the ./demo_vids/ directory.

    Returns:
        list[tf.Tensor]: A list of tensors of shape (batch_size, HEIGHT, WIDTH, 3).
    '''
    imgs = []
    filenames = [
        './demo_imgs/gripper_far_from_grasp.png',
        './demo_imgs/gripper_mid_to_grasp.png',
        './demo_imgs/gripper_almost_grasp.png'
    ]
    for fn in filenames:
        img = Image.open(fn)
        img = np.array(img.resize((WIDTH, HEIGHT)).convert('RGB'))
        img = np.expand_dims(img, axis=0)
        img = tf.reshape(tf.convert_to_tensor(img, dtype=tf.uint8),
                         (1, HEIGHT, WIDTH, 3))
        imgs.append(img)
    return tf.concat(imgs, 0)


def inference(instructions: list[str] | str,
              imgs: list[np.ndarray] | np.ndarray,
              reward: list[float] | float = None,
              policy: LoadedPolicy = None,
              state=None,
              verbose: bool = False,
              step: int = 0,
              done: bool = False):
    '''Runs inference on a list of images and instructions.

    Args:
        instructions (list[str]): A list of instructions. E.g. ["pick up the block"]
        imgs (list[np.ndarray]): A list of images with shape[(HEIGHT, WIDTH, 3)]
        reward (list[float], optional): Defaults to None.
        policy (tf_agents.policies.tf_policy.TFPolicy, optional): Defaults to None.
        state (_type_, optional). Defaults to None.
        verbose (bool, optional): Whether or not to print debugging information. Defaults to False.

    Returns:
        _type_: _description_
    '''
    if policy is None:
        policy = load_rt1()

    if isinstance(instructions, str):
        batch_size = 1
        imgs = np.expand_dims(imgs, axis=0)
        if reward is not None:
            reward = [reward]
    else:
        batch_size = len(instructions)

    reward = tf.constant(reward, dtype=tf.float32)
    imgs = tf.constant(imgs, dtype=tf.uint8)

    if state is None:
        state = policy.get_initial_state(batch_size)
    if reward is None:
        reward = tf.zeros((batch_size,), dtype=tf.float32)

    # Start with zeros in all fields since actual spec has a lot more state
    # than the policy uses.
    observation = specs.zero_spec_nest(specs.from_spec(
        policy.time_step_spec.observation),
                                       outer_dims=(batch_size,))
    observation['image'] = imgs
    observation['natural_language_embedding'] = embed_text(
        instructions, batch_size)
    if step == 0:
        time_step = ts.restart(observation, batch_size)
    elif done:
        time_step = ts.termination(observation, reward)
    else:
        time_step = ts.transition(observation, reward)
    action, next_state, info = policy.action(time_step, state)

    if verbose:
        writer = tf.summary.create_file_writer("logs")
        with writer.as_default():
            for i in range(3):
                tf.summary.scalar('world_vector{}'.format(i),
                                  action['world_vector'][0, i],
                                  step=step)
                tf.summary.scalar('rotation_delta{}'.format(i),
                                  action['rotation_delta'][0, i],
                                  step=step)
            tf.summary.scalar('gripper_closedness_action{}'.format(i),
                              action['gripper_closedness_action'][0, 0],
                              step=step)
            writer.flush()
    return action, next_state


def run_on_demo_imgs(policy: LoadedPolicy = None, verbose: bool = False):
    instructions = "pick block"
    imgs = get_demo_imgs()
    rewards = [0, 0.5, 0.9]
    state = None

    for i in range(3):
        Image.fromarray(imgs[i].numpy().astype(np.uint8)).save(
            'demo_out/test_{}.png'.format(i))
        action, state = inference(instructions,
                                  imgs[i],
                                  rewards[i],
                                  policy,
                                  state,
                                  verbose=True,
                                  step=i,
                                  done=(i == 2))
        pprint(action)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_key',
                        type=str,
                        choices=REGISTRY.keys(),
                        default='xlatest',
                        help='Which model to load.')
    parser.add_argument('-c',
                        '--checkpoint_path',
                        type=str,
                        default=None,
                        help='Custom checkpoint path.')
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='Whether or not to print debugging information.')
    args = parser.parse_args()
    if args.verbose:
        tf.debugging.experimental.enable_dump_debug_info(
            './logs', tensor_debug_mode='FULL_HEALTH')

    run_on_demo_imgs(load_rt1(args.model_key, args.checkpoint_path),
                     verbose=args.verbose)
