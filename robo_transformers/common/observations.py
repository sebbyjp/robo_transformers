from robo_transformers.interface import Observation, Sample
from gym import spaces
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Image(Sample):
    '''Sample for an image.
    '''
    height: int = 480
    width: int = 640
    pixels: np.array = field(default_factory=lambda: np.zeros((480, 640, 3), dtype=np.uint8))

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
