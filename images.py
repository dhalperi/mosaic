import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist

SIZE = 36
MODE = 'L'  # L = grayscale
# Metric is tough, but looks like L1 is better according to at least one study.
METRIC = 'correlation'


def metric_post(dist):
    if METRIC == 'cityblock':
        return dist / SIZE
    elif METRIC == 'correlation':
        return dist * 2**14
    return dist


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
    """Read all the thumbnails of the specified size in the given directory.

    Each image is returned twice, once flipped horizontally."""
    thumbs = {}
    for file in os.listdir('thumbs'):
        if not file.endswith(f'.{SIZE}'):
            continue
        try:
            im = Image.open(f'thumbs/{file}').convert(MODE)
            if im.size != (SIZE, SIZE):
                print(f'Image {file} has size {im.size} and is_animated: {im.is_animated}')
                continue
            thumbs[file] = np.array([b for b in im.tobytes()], dtype=np.uint8)
            thumbs[file + '.flip'] = np.array([b for b in im.transpose(Image.FLIP_LEFT_RIGHT).tobytes()],
                                              dtype=np.uint8)
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
        if count & 0xFFFFF == 0:
            print(count, len(used))
        i = int(i)
        j = int(j)
        idx = i & 0xFFFFFFFE  # mask out bottom bit to indicate flipping
        if idx not in used and j not in matches:
            matches[j] = i
            used.add(idx)
            if len(matches) >= break_at:
                break
    return matches


if __name__ == "__main__":
    print('Reading and slicing target')
    target = Image.open('IMG_3178.JPG').convert(MODE)
    slices = slice_target(target)
    slices_offsets = slices[0]
    slices_data = slices[1]
    slices_matrix = np.array(slices_data, dtype=np.uint8)
    print(f'Produced {len(slices_offsets)} slices and a {slices_matrix.shape} array of their bytes')

    print('Reading thumbnails')
    thumbs = list(read_thumbs().items())  # type: List[Tuple[str, np.ndarray]]
    thumbs_matrix = np.array([v for (k, v) in thumbs], dtype=np.uint8)
    print(f'Read {len(thumbs)} thumbnails of the right size and produced a {thumbs_matrix.shape} array of their bytes')

    if not os.path.exists(f'dist-{METRIC}.npy'):
        print('Computing distances')
        dist = metric_post(cdist(thumbs_matrix, slices_matrix, METRIC)).astype(np.uint16)
        print('Computed a', dist.shape, 'array of distances')
        np.save(f'dist-{METRIC}.npy', dist)
    else:
        dist = np.load(f'dist-{METRIC}.npy')

    if not os.path.exists(f'matches-{METRIC}.out'):
        matches = compute_matches_greedy_matching(dist)
        with open(f'matches-{METRIC}.out', 'w') as outfile:
            json.dump(matches, outfile)
    else:
        with open(f'matches-{METRIC}.out', 'r') as infile:
            matches = {int(i): j for i, j in json.load(infile).items()}

    P = 15
    output = Image.new(MODE, target.size)
    for i, j in matches.items():
        (x, y) = slices_offsets[i]
        slice = slices_data[i]
        match = thumbs[j][1].astype(np.int16)
        pct = np.percentile(match, [P, 100-P])
        shift = (np.mean(slice) - np.mean(match))
        match += int(min(max(-pct[0], shift), 255-pct[1]))
        match = match.clip(0, 255).astype(np.uint8)
        output.paste(Image.frombytes(MODE, (SIZE, SIZE), match), (x, y, x + SIZE, y + SIZE))
    output.save(f'output/output-{MODE}-{METRIC}-shiftpct-{P}.jpg')
