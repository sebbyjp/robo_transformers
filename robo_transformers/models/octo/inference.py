import os
from octo.model.octo_model import OctoModel
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import jax

model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small")


from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np
# download one example BridgeV2 image
IMAGE_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/raw/bridge_data_v2/datacol2_toykitchen7/drawer_pnp/01/2023-04-19_09-18-15/raw/traj_group0/traj0/images0/im_12.jpg"
img = np.array(Image.open(requests.get(IMAGE_URL, stream=True).raw).resize((256, 256)))
plt.imshow(img)


# add batch + time horizon 1
img = img[np.newaxis,np.newaxis,...]
observation = {"image_primary": img, "pad_mask": np.array([[True]])}
task = model.create_tasks(texts=["pick up the fork"])
print(model.module.tabulate(jax.random.PRNGKey(0),observation, task, observation['pad_mask'],train=False, verbose=True, depth=2)
            )  # Prints out the parameter count of our model, adn tokenizer details
action = model.sample_actions(observation, task, rng=jax.random.PRNGKey(0))
print(action)   # [batch, action_chunk, action_dim]
print(model.get_pretty_spec())