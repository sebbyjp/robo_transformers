import tensorflow as tf
from pprint import pprint
import numpy as np
import PIL.Image as Image
import tensorflow_hub as hub
from tf_agents.policies.py_tf_eager_policy import SavedModelPyTFEagerPolicy as LoadedPolicy
from tf_agents.trajectories import time_step as ts
import tf_agents
import absl.logging

WIDTH = 320
HEIGHT = 256
TEXT_ENCODER = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def load_rt1(
        checkpoint_path:
    str = './third_party/rt1/robotics_transformer/trained_checkpoints/rt1main',
        # rt_1_x_tf_trained_for_002272480_step"
        load_specs_from_pbtxt: bool = True,
        use_tf_function: bool = True,
        batch_time_steps: bool = False,
        verbose: bool = False) -> tf_agents.policies.tf_policy.TFPolicy:
    '''Loads a trained RT-1 model from a checkpoint.

    Args:
        checkpoint_path (str, optional): Defaults to './robotics_transformer/trained_checkpoints/rt1main'.
        load_specs_from_pbtxt (bool, optional): Load from the .pb file in the checkpoint dir. Defaults to True.
        use_tf_function (bool, optional): Wraps function with optimized tf.Function to speed up inference. Defaults to True.
        batch_time_steps (bool, optional): Whether to automatically add a batch dimension during inference. Defaults to False.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        tf_agents.polices.tf_policy.TFPolicy: A tf_agents policy object.
    '''
    if not verbose:
        absl.logging.set_verbosity(absl.logging.ERROR)
    print('Loading RT-1 from checkpoint: {}...'.format(checkpoint_path))
    rt1_policy: LoadedPolicy = LoadedPolicy(
        model_path=checkpoint_path,
        load_specs_from_pbtxt=load_specs_from_pbtxt,
        use_tf_function=use_tf_function,
        batch_time_steps=batch_time_steps)
    absl.logging.set_verbosity(absl.logging.WARN)
    print('RT-1 loaded.')
    return rt1_policy


def embed_text(input: str, batch_size: int = 1) -> tf.Tensor:
    '''Embeds a string using the Universal Sentence Encoder.

    Args:
        input (str): The string to embed.
        batch_size (int, optional): . Defaults to 1.

    Returns:
        tf.Tensor: A tensor of shape (batch_size, 512).
    '''
    embedded = TEXT_ENCODER(input).numpy()[0]
    return tf.reshape(tf.convert_to_tensor(embedded, dtype=tf.float32),
                      (batch_size, 512))


def get_demo_imgs(batch_size: int = 1) -> list[tf.Tensor]:
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
        img = tf.reshape(tf.convert_to_tensor(img, dtype=tf.uint8),
                         (batch_size, HEIGHT, WIDTH, 3))
        imgs.append(img)
    return imgs


# Dumbed down representations of the actual specs.


class RT1InternalState(object):

    def __init__(self,
                 batch_size: int = 1,
                 nframes: int = 6,
                 imgs: np.ndarray = None,
                 t: int = 5,
                 step_num: int = 5):
        if imgs is None:
            self.imgs = tf.zeros((batch_size, 6, HEIGHT, WIDTH, 3),
                                 dtype=np.uint8)
        self.batch_size = batch_size
        self.nframes = nframes
        self.t = tf.constant(t,
                             shape=(batch_size, nframes, 1, 1, 1, 1),
                             dtype=tf.int32)
        self.step_num = tf.constant(step_num,
                                    shape=(batch_size, nframes, 1, 1, 1, 1),
                                    dtype=tf.int32)


class RT1InferenceObservation(object):

    def __init__(self,
                 img: np.ndarray = None,
                 instructions: str = "pick up the block"):
        if img is None:
            self.imgs = tf.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def inference(instructions: list[str],
              imgs: np.ndarray,
              batch_size: int = 1,
              reward: np.ndarray = None,
              rt1_policy: tf_agents.policies.tf_policy.TFPolicy = None,
              rt1_curr_state=None):
    # Actually just run from the beginning instead of inventing state if we don't have it.
    if rt1_policy is None:
        rt1_policy = load_rt1()
    if rt1_curr_state is None:
        state = rt1_policy.get_initial_state(batch_size)
        # Pretend that we have some state from a previous timestep.
        state['image'] = tf.tile(tf.reshape(imgs[0], (1, 1, HEIGHT, WIDTH, 3)),
                                 [1, 6, 1, 1, 1])
        # Pretend this is timestep 5 out of 6.
        state['t'] = tf.constant(5,
                                 shape=(batch_size, 1, 1, 1, 1),
                                 dtype=tf.int32)
        state['step_num'] = tf.constant(5,
                                        shape=(batch_size, 1, 1, 1, 1),
                                        dtype=tf.int32)

    # Start with zeros in all fields since actual spec has a lot more state
    # than we care to give.
    observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(
        rt1_policy.time_step_spec.observation),
                                                 outer_dims=(batch_size,))
    observation['image'] = imgs[-1]
    observation['natural_language_embedding'] = instructions[-1]

    time_step = ts.transition(observation,
                              reward=reward * tf.ones(
                                  (batch_size,), dtype=tf.float32),
                              outer_dims=(batch_size,))

    action, state, _ = rt1_policy.action(time_step, state)

    # View the results.
    pprint(action)
    # TODO(speralta): convert this to the dummed down specs.
    return action, state


# TODO(speralta): Use the inference method above to run on the demo images.
def run_on_demo_imgs(batch_size: int = 1,
                     rt1_policy: tf_agents.policies.tf_policy.TFPolicy = None):
    instructions = embed_text(["pick up the block"], batch_size)
    imgs = get_demo_imgs(batch_size)

    observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(
        rt1_policy.time_step_spec.observation),
                                                 outer_dims=(batch_size,))

    # Give a small reward for the first observation which is far from the block.
    reward = 0.25
    observation['image'] = imgs[0]
    observation['natural_language_embedding'] = instructions

    state = rt1_policy.get_initial_state(batch_size)

    # Simulate the environment by settting the state.
    state['image'] = tf.tile(tf.reshape(imgs[0], (1, 1, HEIGHT, WIDTH, 3)),
                             [1, 6, 1, 1, 1])

    # Pretend this is timestep 5 out of 6.
    state['t'] = tf.constant(5, shape=(batch_size, 1, 1, 1, 1), dtype=tf.int32)
    state['step_num'] = tf.constant(5,
                                    shape=(batch_size, 1, 1, 1, 1),
                                    dtype=tf.int32)
    time_step = ts.restart(observation, batch_size)
    for i in range(6):
        # Simulate the first image, far from the block.
        action, state, _ = rt1_policy.action(time_step, state)
        time_step = ts.transition(observation,
                                  reward=reward * tf.ones(
                                      (batch_size,), dtype=tf.float32),
                                  outer_dims=(batch_size,))

        # Simulate the second image, closer to the block.
        if i > 1 and i < 3:
            observation['image'] = imgs[1]
            reward = 0.5
        elif i >= 3:
            observation['image'] = imgs[2]
            reward = 0.8

    # View the results.
    pprint(action)
    for i in range(6):
        img_out = Image.fromarray(state['image'][0, i, :, :, :].numpy().astype(
            np.uint8))
        img_out.save('demo_out/test_{}.png'.format(i))


if __name__ == '__main__':

    run_on_demo_imgs()
