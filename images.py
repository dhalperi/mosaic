import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist, cityblock

from lib import read_thumbs

DIFF_SIZE = 36
TILE_SIZE = 64
MODE = 'L'  # L = grayscale
# Metric is tough.
# L2 is "standard", but looks like L1 is better according to at least one study.
# L1 does seem to look better in practice, but correlation (+shifting output) seems even better.
# Seems like there ought to be another opportunity here -- since correlation overestimates
#    ability to shift (w/clipping), but the L1-shifted implementation below seems not to work.
#    ... and it's REALLY slow.
METRIC = 'correlation'


def compute_dist(thumbs_matrix: np.ndarray, slices_matrix: np.ndarray) -> np.ndarray:
    if METRIC == 'cityblock':
        raw_metric = cdist(thumbs_matrix, slices_matrix, 'cityblock')
        return (raw_metric / DIFF_SIZE).astype(np.uint16)
    elif METRIC == 'correlation':
        raw_metric = cdist(thumbs_matrix, slices_matrix, 'correlation')
        return (raw_metric * 2 ** 14).astype(np.uint16)
    elif METRIC == 'euclidean':
        raw_metric = cdist(thumbs_matrix, slices_matrix, 'euclidean')
        return raw_metric.astype(np.uint16)
    elif METRIC == 'L1-shifted':
        P = 15
        mean_slices = np.mean(slices_matrix, 1).astype(np.int16)
        M = thumbs_matrix.shape[0]
        N = slices_matrix.shape[0]
        ret = np.ndarray((M, N), np.uint16)
        for i in range(M):
            thumb = thumbs_matrix[i].astype(np.int16)
            m = np.mean(thumb)
            pct = np.percentile(thumb, [P, 100 - P])
            lower = int(-pct[0])
            upper = int(255 - pct[1])
            for j in range(N):
                shift_thumb = thumb + min(max(lower, mean_slices[j] - m), upper)
                ret[i, j] = cityblock(slices_matrix[j], shift_thumb.clip(0, 255).astype(np.uint8))

        return ret

    raise ValueError(f'Unknown metric: {METRIC}')


def produce_output_tile(thumb, target_tile, x, y):
    if METRIC in ['correlation', 'L1-shifted']:
        thumb = thumb.astype(np.int16)
        P = 15
        pct = np.percentile(thumb, [P, 100 - P])
        shift = (np.mean(target_tile) - np.mean(thumb))
        thumb += int(min(max(-pct[0], shift), 255 - pct[1]))
        thumb = thumb.clip(0, 255).astype(np.uint8)
    return thumb


def slice_target(im: Image) -> Tuple[List[Tuple[int, int]], List[np.ndarray]]:
    """Break the target image into SIZE x SIZE tiles and return a list of numpy arrays of their contents"""
    positions = []
    data = []
    (w, h) = im.size
    for i in range(0, w // DIFF_SIZE):
        for j in range(0, h // DIFF_SIZE):
            positions.append((i, j))
            pi = i * DIFF_SIZE
            pj = j * DIFF_SIZE
            data.append(
                np.array([b for b in im.crop((pi, pj, pi + DIFF_SIZE, pj + DIFF_SIZE)).tobytes()], dtype=np.uint8))

    return positions, data


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

    # Shrink the input distance array by two, picking the better for each of the mirrors.
    L, T = dist.shape
    dist_shrunk = np.amin(np.resize(dist, (L // 2, 2, T)), 1)
    print(dist_shrunk.shape)

    print('Reshaping dist into a 1-D array')
    dist_1d = np.reshape(dist_shrunk, (dist_shrunk.size, 1))

    print('Sorting dist')
    min_idx_1d = np.argsort(dist_1d, axis=0, kind='stable')
    min_idx = np.array(np.unravel_index(min_idx_1d, dist_shrunk.shape)).T[0]

    print('Matching')
    count = 0
    matches = {}
    used = set()
    break_at = min(dist_shrunk.shape)
    for (i, j) in min_idx:
        count += 1
        if count & 0xFFFFF == 0:
            print(count, len(matches))
        j = int(j)
        if j in matches:
            continue
        i = int(i)
        if i in used:
            continue
        if dist_shrunk[i, j] == dist[2 * i, j]:
            matches[j] = 2 * i
        else:
            assert dist_shrunk[i, j] == dist[2 * i + 1, j]
            matches[j] = 2 * i + 1
        used.add(i)
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
    thumbs = read_thumbs(size=TILE_SIZE, mode=MODE, include_flips=True, resize=DIFF_SIZE)
    thumbs_matrix = np.array([t.bytes for t in thumbs], dtype=np.uint8)
    print(f'Read {len(thumbs)} thumbnails of the right size and produced a {thumbs_matrix.shape} array of their bytes')

    # Compute distance based on the given metric, or load cached distance
    if not os.path.exists(f'dist-{METRIC}.npy'):
        print('Computing distances')
        dist = compute_dist(thumbs_matrix, slices_matrix)
        print('Computed a', dist.shape, 'array of distances')
        np.save(f'dist-{METRIC}.npy', dist)
    else:
        dist = np.load(f'dist-{METRIC}.npy')

    # Compute matches, or load cached matches
    if not os.path.exists(f'matches-{METRIC}.out'):
        matches = compute_matches_greedy_matching(dist)
        with open(f'matches-{METRIC}.out', 'w') as outfile:
            json.dump(matches, outfile)
    else:
        with open(f'matches-{METRIC}.out', 'r') as infile:
            matches = {int(i): j for i, j in json.load(infile).items()}

    # Assemble the mosaic from all the chosen tiles
    print('Assembling mosaic')
    output_size = (target.size[0] // DIFF_SIZE * TILE_SIZE, target.size[1] // DIFF_SIZE * TILE_SIZE)
    mosaic = Image.new(MODE, output_size)
    for i, j in matches.items():
        (x, y) = slices_offsets[i]
        thumb = thumbs[j]
        data = np.array([b for b in Image.open(f'thumbs/{thumb.uid}.{TILE_SIZE}').convert(MODE).tobytes()])
        match = produce_output_tile(data, slices_data[i], x, y)
        px = x * TILE_SIZE
        py = y * TILE_SIZE
        mosaic.paste(Image.frombytes(MODE, (TILE_SIZE, TILE_SIZE), match),
                     (px, py, px + TILE_SIZE, py + TILE_SIZE))

    # Do a slight blending of assembled mosaic and target image to make it look better
    print('Blending mosaic and target')
    alpha = 15
    if target.size != output_size:
        target = target.resize(output_size)
    blended = Image.blend(mosaic, target, alpha / 100)

    print('Saving produced mosaic')
    blended.save(f'output/output-{MODE}-{METRIC}.jpg')
