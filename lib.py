import os
from typing import List, NamedTuple

import numpy as np
from PIL import Image


class Thumb(NamedTuple):
    uid: str
    bytes: np.ndarray
    flipped: bool


def read_thumbs(size, mode='L', include_flips=False) -> List[Thumb]:
    """Read all the thumbnails of the specified size in the given directory.

    Each image is returned twice, once flipped horizontally."""
    thumbs = []
    for file in os.listdir('thumbs'):
        if not file.endswith(f'.{size}'):
            continue
        try:
            im = Image.open(f'thumbs/{file}').convert(mode)
            if im.size != (size, size):
                print(f'Image {file} has size {im.size} and is_animated: {im.is_animated}')
                continue
            fname = os.path.splitext(os.path.basename(file))[0]
            thumbs.append(Thumb(uid=fname, bytes=np.frombuffer(im.tobytes(), dtype=np.uint8), flipped=False))
            if include_flips:
                thumbs.append(
                    Thumb(uid=fname,
                          bytes=np.frombuffer(im.transpose(Image.FLIP_LEFT_RIGHT).tobytes(), dtype=np.uint8),
                          flipped=True))
        except IOError:
            pass

    return thumbs
