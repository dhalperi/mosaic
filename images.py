import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

SIZE = 36
MODE = 'L'  # L = grayscale


def slice_target(im: Image) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    """Break the target image into SIZE x SIZE tiles and return a list of numpy arrays of their contents"""
    positions = []
    data = []
    (w, h) = im.size
    for i in range(0, w, SIZE):
        for j in range(0, h, SIZE):
            positions.append((i, j))
            data.append(np.array([b for b in im.crop((i, j, i + SIZE, j + SIZE)).tobytes()], dtype=np.uint8))

    return positions, data


def read_thumbs() -> Dict[str, np.ndarray]:
    """Read all the thumbnails of the specified size in the given directory"""
    thumbs = {}
    for file in os.listdir('thumbs'):
        if not file.endswith(f'.{SIZE}'):
            continue
        try:
            im = Image.open(f'thumbs/{file}').convert(MODE)
            if im.size != (SIZE, SIZE):
                continue
            thumbs[file] = np.array([b for b in im.tobytes()], dtype=np.uint8)
        except IOError:
            pass
    return thumbs


def delta(b1: bytes, b2: bytes) -> int:
    assert len(b1) == len(b2), f'{len(b1)} != {len(b2)}'
    return sum(abs(a - b) * abs(a - b) for a, b in zip(b1, b2))


def match(slice: bytes, thumbs: Dict[str, bytes]) -> str:
    assert thumbs
    curid = None
    cur = None

    for (id, data) in thumbs.items():
        try:
            d = delta(slice, data)
            if cur is None or d < cur:
                curid = id
                cur = d
        except AssertionError:
            pass
    return curid


def compute_matches_best_repeats(dist: np.ndarray) -> Dict[int, int]:
    """Computes a trivial matching by picking the thumbnail that best matches each tile, with repeats"""
    best_matches = np.argmin(dist, axis=0)
    return dict(enumerate(best_matches))


def compute_matches_greedy_matching(dist: np.ndarray) -> Dict[int, int]:
    """Computes a greedy matching with no repeats.

    Iteratively picks the smallest entries in the distance array that correspond to unmatched rows and columns."""

    print('Reshaping dist into a 1-D array')
    dist_1d = np.reshape(dist, (dist.size, 1))

    print('Sorting dist')
    min_idx_1d = np.argsort(dist_1d, axis=0)
    min_idx = np.array(np.unravel_index(min_idx_1d, dist.shape)).T[0]

    print('Matching')
    count = 0
    matches = {}
    used = set()
    break_at = min(dist.shape)
    for (i, j) in min_idx:
        count += 1
        if count & 0xFFFF == 0:
            print(count, len(used))
        i = int(i)
        j = int(j)
        if i not in used and j not in matches:
            matches[j] = i
            used.add(i)
            if len(matches) >= break_at:
                break
    return matches


if __name__ == "__main__":
    print('Reading and slicing target')
    target = Image.open('IMG_3178.JPG').convert(MODE)
    slices = slice_target(target)
    slices_offsets = slices[0]
    slices_matrix = np.array(slices[1], dtype=np.uint8)
    print(f'Produced {len(slices_offsets)} slices and a {slices_matrix.shape} array of their bytes')

    print('Reading thumbnails')
    thumbs = list(read_thumbs().items())  # type: List[Tuple[str, np.ndarray]]
    thumbs_matrix = np.array([v for (k, v) in thumbs], dtype=np.uint8)
    print(f'Read {len(thumbs)} thumbnails of the right size and produced a {thumbs_matrix.shape} array of their bytes')

    if not os.path.exists('dist.npy'):
        print('Computing distances')
        dist = cdist(thumbs_matrix, slices_matrix, 'euclidean').astype(np.uint16)
        print('Computed a', dist.shape, 'array of distances')
        np.save('dist.npy', dist)
    else:
        dist = np.load('dist.npy')

    if not os.path.exists('matches.out'):
        matches = compute_matches_greedy_matching(dist)
        with open('matches.out', 'w') as outfile:
            json.dump(matches, outfile)
    else:
        with open('matches.out', 'r') as infile:
            matches = json.load(infile)

    output = Image.new(MODE, target.size)
    for i, j in matches.items():
        (x, y) = slices_offsets[i]
        match = thumbs[j]
        output.paste(Image.frombytes(MODE, (SIZE, SIZE), match[1]), (x, y, x + SIZE, y + SIZE))
    output.save(f'output/output-{MODE}.jpg')
