from PIL import Image
import numpy as np
from typing import Optional
from pprint import pprint
from absl import logging
from beartype import beartype
from numpy.typing import ArrayLike
from robo_transformers.registry import REGISTRY
from robo_transformers.interface import Agent

@beartype
class InferenceServer:

    def __init__(self,
                 model_uri: str = "gs://rt1/rt1main",
                 agent: Optional[Agent] = None,
                 **kwargs
                 ):
        '''Initializes the inference server.


        Args:
            model_uri (str, optional): The URI of the model. Defaults to "gs://rt1/rt1main".
            agent (Optional[Agent], optional): The agent to use. Defaults to None.
            **kwargs: kwargs for the agent.

        '''

        if agent is not None:
            self.agent: Agent = agent
        else:
            model_type = model_uri.split('/')[-2]
            model_variant = model_uri.split('/')[-1]
            self.agent: Agent = REGISTRY[model_type]['agent'](model_uri, **kwargs)
            self.action = None

    def __call__(self,
                 save: bool = False,
                 **kwargs
                 ) -> list[dict]:
        '''Runs inference on a Vision Language Action model.


        Args:
            save (bool, optional): Whether or not to save the observed image. Defaults to False.
            *args: args for the agent.
            **kwargs: kwargs for the agent.

        Returns:
            dict: See RT1Action for details.
        '''
        image: ArrayLike = kwargs.get('image')
        if image is not None and save:
            Image.fromarray(np.array(image, dtype=np.uint8)).save("rt1_saved_image.png")

        try:
            self.action = self.agent.act(**kwargs)

            if logging.get_verbosity() > logging.DEBUG and kwargs.get('instruction') is not None:
                intruction = kwargs['instruction']
                print(f'instruction: {intruction}')
                pprint(self.action)

        except Exception as e:
            import traceback
            traceback.print_tb(e.__traceback__)
            raise e

        return self.action

