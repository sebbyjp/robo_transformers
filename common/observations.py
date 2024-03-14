from robo_transformers.interface import Observation, Sample
from gym import spaces
import numpy as np
from dataclasses import dataclass, field
from PIL import Image as PILImage
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640

@dataclass
class Image(Sample):
    '''Sample for an image.
    '''
    height: int = DEFAULT_HEIGHT
    width: int = DEFAULT_WIDTH
    pixels: np.array = field(default_factory=lambda: np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8))

    def __post_init__(self):
        self.pixels = PILImage.fromarray(self.pixels).resize((self.width, self.height))

    def space(self) -> spaces.Dict:
        return spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)


@dataclass
class ImageInstruction(Observation):
    '''Sample for a set of instruction images.
    '''
    instruction: str = " "
    image: Image = field(default_factory=Image)

    def space(self) -> spaces.Dict:
        space = dict(super().space())
        space.update({
            "instruction": spaces.Text(100),
            "image": self.image.space()
        })
        return spaces.Dict(space)

@dataclass
class MultiImageInstruction(Observation):
    '''Sample for a set of instruction images.
    '''
    instruction: str = " "
    images: list = field(default_factory=lambda: [Image(), Image()])

    def space(self) -> spaces.Dict:
        return spaces.Dict({
            "instruction": spaces.Text(100),
            "images": spaces.Tuple([image.space() for image in self.images])
        })
