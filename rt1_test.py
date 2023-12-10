import tensorflow as tf
from robotics_transformer.sequence_agent import SequenceAgent
from tensor2robot.utils import tensorspec_utils
from tf_agents.specs import tensor_spec
from pprint import pprint
import numpy as np
import PIL.Image as Image
import PIL
import tensorflow_hub as hub
from tf_agents.policies.py_tf_eager_policy import SavedModelPyTFEagerPolicy as LoadedPolicy
from tf_agents.trajectories import time_step as ts
import tf_agents
import tf_agents.utils.nest_utils as nest_utils
from tf_agents.agents import data_converter
import absl.logging

WIDTH = 320
HEIGHT = 256
def load_model(path= '/Users/sebastianperalta/simply/dev/rt_test/robotics_transformer/trained_checkpoints/rt1main'):
    absl.logging.set_verbosity(absl.logging.ERROR)
    # path="/Users/sebastianperalta/simply/dev/rt_test/rt_1_x_tf_trained_for_002272480_step"
    tfa_policy: LoadedPolicy = LoadedPolicy(
        model_path= path,
        load_specs_from_pbtxt=True,
        use_tf_function=True,
        batch_time_steps=False)
    absl.logging.set_verbosity(absl.logging.WARN)
    return tfa_policy

use_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def embed(input):
  return use_encoder(input)


img = Image.open('/Users/sebastianperalta/simply/media/block_demo_real.png').resize((WIDTH, HEIGHT)).convert('RGB')
img = np.array(img)
embedded = embed(["pick up the block"]).numpy()[0]

# Perform one step of inference using dummy input
tfa_policy = load_model()
# tfa_policy = SequenceAgent(tfa_policy)
# Obtain a dummy observation, where the features are all 0
observation = tf_agents.specs.zero_spec_nest(tf_agents.specs.from_spec(tfa_policy.time_step_spec.observation), outer_dims=(1,))
observation['image'] = tf.reshape(tf.convert_to_tensor(img, dtype=tf.uint8), (1,HEIGHT,WIDTH,3))
observation['natural_language_embedding'] = tf.reshape(tf.convert_to_tensor(embedded, dtype=tf.float32), (1,512))

# Construct a tf_agents time_step from the dummy observation
# steps = ts.transition(observation, reward=tf_agents.specs.zero_spec_nest((1,6)), outer_dims=(6,))

step = ts.restart(observation, batch_size=1)
state = tfa_policy.get_initial_state(batch_size=1)
state['image'] =  tf.tile(tf.reshape(img, (1,1,HEIGHT, WIDTH, 3)), [1,6,1,1,1])
state['t'] = tf.constant(5, shape=(1,1,1,1,1), dtype=tf.int32)
state['step_num'] = tf.constant(5, shape=(1,1,1,1,1), dtype=tf.int32)
for i in range(6):
    action, state, _ = tfa_policy.action(step, state)
    step = ts.transition(observation, reward=tf.ones((1,), dtype=tf.float32), outer_dims=(1,))
    
# time_steps, policy_steps, _ = tfa_policy.as_transition(observation)
# policy_state = tfa_policy.get_initial_state(batch_size=1)
# actions = tfa_policy.action(time_steps, policy_state=policy_state)
# data_context = data_converter.DataContext(
#     time_step_spec=tfa_policy.time_step_spec,
#     action_spec=tfa_policy.action_spec,
#     info_spec=tfa_policy.info_spec,
#     use_half_transition=True)
# as_transition = data_converter.AsHalfTransition(
#     data_context, squeeze_time_dim=False)
# transition = as_transition(observation)
# time_steps, policy_steps, _ = transition
# batch_size = nest_utils.get_outer_shape(time_steps,tfa_policy._time_step_spec)[0]
# tfa_policy.set_actions(policy_steps.action)
# policy_state = tfa_policy.get_initial_state(batch_size)
# actions = tfa_policy.action(time_steps, policy_state=policy_state)
# Initialize the state of the policy

# policy_state['t'] = tf.constant(5, shape=(1,1,1,1), dtype=tf.int32)
# policy_state['step_num'] = tf.constant(5, shape=(1,1,1,1), dtype=tf.int32)
# Run inference using the policy

pprint(action)
for i in range(6):
    img_out = Image.fromarray(state['image'][0,i,:,:,:].numpy().astype(np.uint8))
    img_out.save('test_imgs/test_{}.png'.format(i))




# Rest of your code...
