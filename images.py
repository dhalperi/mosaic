import json
import logging
import os
import time
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist, cityblock

from lib import read_thumbs

SLICE_SIZE = 28
DIFF_SIZE = 28
TILE_SIZE = 128
MODE = "L"  # L = grayscale
# Metric is tough.
# L2 is "standard", but looks like L1 is better according to at least one study.
# L1 does seem to look better in practice, but correlation (+shifting output) seems even better.
# Correlation regularized by L2 seems best so far - it penalizes the amount of shifting needed.
METRIC = "correlation-regularized"


def compute_dist(thumbs_matrix: np.ndarray, slices_matrix: np.ndarray) -> np.ndarray:
    # Break each array in half, so that we can run 4 computations at once
    with Pool(processes=4) as pool:
        T = thumbs_matrix.shape[0]
        S = slices_matrix.shape[0]
        t1 = thumbs_matrix[: T // 2, :]
        t2 = thumbs_matrix[T // 2 :, :]
        s1 = slices_matrix[: S // 2, :]
        s2 = slices_matrix[S // 2 :, :]
        d11, d12, d21, d22 = pool.starmap(
            compute_dist_internal, ((t1, s1), (t1, s2), (t2, s1), (t2, s2))
        )
        return np.concatenate(
            [np.concatenate([d11, d12], axis=1), np.concatenate([d21, d22], axis=1)],
            axis=0,
        )


def compute_dist_internal(
    thumbs_matrix: np.ndarray, slices_matrix: np.ndarray
) -> np.ndarray:
    if METRIC == "cityblock":
        raw_metric = cdist(thumbs_matrix, slices_matrix, "cityblock")
        return (raw_metric / DIFF_SIZE).astype(np.uint16)
    elif METRIC == "correlation":
        raw_metric = cdist(thumbs_matrix, slices_matrix, "correlation")
        return (raw_metric * 2 ** 14).astype(np.uint16)
    elif METRIC == "correlation-regularized":
        mean_thumb = np.mean(thumbs_matrix, 1).reshape((-1, 1))
        mean_slice = np.mean(slices_matrix, 1).reshape((-1, 1))
        mean = (cdist(mean_thumb, mean_slice, "sqeuclidean") / 2).astype(np.uint16)
        corr = (cdist(thumbs_matrix, slices_matrix, "correlation") * 2 ** 14).astype(
            np.uint16
        )
        ret = corr + mean
        return ret
    elif METRIC == "euclidean":
        raw_metric = cdist(thumbs_matrix, slices_matrix, "euclidean")
        return raw_metric.astype(np.uint16)
    elif METRIC == "L1-shifted":
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
                ret[i, j] = cityblock(
                    slices_matrix[j], shift_thumb.clip(0, 255).astype(np.uint8)
                )

        return ret

    raise ValueError(f"Unknown metric: {METRIC}")


def produce_output_tile(thumb, target_tile, x, y):
    if METRIC in ["correlation", "correlation-regularized", "L1-shifted"]:
        thumb = thumb.astype(np.int16)
        P = 15
        pct = np.percentile(thumb, [P, 100 - P])
        shift = np.mean(target_tile) - np.mean(thumb)
        thumb += int(min(max(-pct[0], shift), 255 - pct[1]))
        thumb = thumb.clip(0, 255).astype(np.uint8)
    return thumb


def slice_target(
    im: Image,
) -> Tuple[List[Tuple[int, int]], List[np.ndarray], List[np.ndarray]]:
    """Break the target image into SIZE x SIZE tiles and return a list of numpy arrays of their contents"""
    positions = []
    data = []
    data_diff = []
    (w, h) = im.size
    for i in range(0, w // SLICE_SIZE):
        for j in range(0, h // SLICE_SIZE):
            positions.append((i, j))
            pi = i * SLICE_SIZE
            pj = j * SLICE_SIZE
            cropped = im.crop((pi, pj, pi + SLICE_SIZE, pj + SLICE_SIZE))
            resized = cropped.resize((DIFF_SIZE, DIFF_SIZE))
            data.append(
                np.array(
                    list(cropped.tobytes()),
                    dtype=np.uint8,
                )
            )
            data_diff.append(
                np.array(
                    list(resized.tobytes()),
                    dtype=np.uint8,
                )
            )

    return positions, data, data_diff


def compute_matches_scipy(dist: np.ndarray) -> Dict[int, int]:
    """Lets scipy compute the minimum cost matching based on the diff array."""

    # Shrink the input distance array by two, picking the better for each of the mirrors.
    logging.info("Picking the better image of mirrored images")
    L, T = dist.shape
    reoriented = np.resize(dist, (L // 2, 2, T))
    best_mirror = np.argmin(reoriented, 1)
    dist_shrunk = np.amin(reoriented, 1)

    import scipy.optimize

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(dist_shrunk)
    return dict(
        (int(j), 2 * int(i) + int(best_mirror[i][j])) for i, j in zip(row_ind, col_ind)
    )


def get_file_create_time(path):
    return time.localtime(os.path.getmtime(path))


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(message)s", level=logging.INFO)
    logging.info("Reading and slicing target")
    target = Image.open("IMG_3178.JPG").convert(MODE)
    slices = slice_target(target)
    slices_offsets = slices[0]
    slices_data = slices[1]
    slices_matrix = np.array(slices_data, dtype=np.uint8)
    slices_diff_matrix = np.array(slices[2], dtype=np.uint8)
    logging.info(
        "Produced %s slices and a %s array of their bytes",
        len(slices_offsets),
        slices_matrix.shape,
    )

    mtime = get_file_create_time("media_items.json")
    logging.info(
        "Reading thumbnails from media_items.json updated at %s", time.asctime(mtime)
    )
    thumbs = read_thumbs(
        size=TILE_SIZE, mode=MODE, include_flips=True, resize=DIFF_SIZE
    )
    thumbs_matrix = np.array([t.bytes for t in thumbs], dtype=np.uint8)
    logging.info(
        "Read %s thumbnails of the right size and produced a %s array of their bytes",
        len(thumbs),
        thumbs_matrix.shape,
    )

    # Compute distance based on the given metric, or load cached distance
    dist_filename = f"dist-{METRIC}-{DIFF_SIZE}-{SLICE_SIZE}.npy"
    if not os.path.exists(dist_filename) or get_file_create_time(dist_filename) < mtime:
        logging.info("Computing distances")
        dist = compute_dist(thumbs_matrix, slices_diff_matrix)
        logging.info("Computed a %s array of distances", dist.shape)
        np.save(dist_filename, dist)
    else:
        logging.info("Loading existing distances from %s", dist_filename)
        dist = np.load(dist_filename)

    # Compute matches, or load cached matches
    matches_filename = f"matches-{METRIC}-{DIFF_SIZE}-{SLICE_SIZE}.out"
    if (
        not os.path.exists(matches_filename)
        or get_file_create_time(matches_filename) < mtime
    ):
        matches = compute_matches_scipy(dist)
        with open(matches_filename, "w") as outfile:
            json.dump(matches, outfile)
    else:
        logging.info("Loading existing matches from %s", matches_filename)
        with open(matches_filename, "r") as infile:
            matches = {int(i): j for i, j in json.load(infile).items()}

    # Assemble the mosaic from all the chosen tiles
    logging.info("Assembling mosaic")
    output_size = (
        target.size[0] // SLICE_SIZE * TILE_SIZE,
        target.size[1] // SLICE_SIZE * TILE_SIZE,
    )
    mosaic = Image.new(MODE, output_size)
    for i, j in matches.items():
        (x, y) = slices_offsets[i]
        thumb = thumbs[j]
        thumb_img = Image.open(f"thumbs/{thumb.uid}.{TILE_SIZE}").convert(MODE)
        if thumb.flipped:
            thumb_img = thumb_img.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.frombuffer(thumb_img.tobytes(), dtype=np.uint8)
        match = produce_output_tile(data, slices_data[i], x, y)
        px = x * TILE_SIZE
        py = y * TILE_SIZE
        mosaic.paste(
            Image.frombytes(MODE, (TILE_SIZE, TILE_SIZE), match),
            (px, py, px + TILE_SIZE, py + TILE_SIZE),
        )

    # Do a slight blending of assembled mosaic and target image to make it look better
    logging.info("Blending mosaic and target")
    alpha = 10
    if target.size != output_size:
        target = target.resize(output_size)
    blended = Image.blend(mosaic, target, alpha / 100)

    if not os.path.exists("output"):
        os.mkdir("output")
    output_filename = f"output/output-{MODE}-{METRIC}-{DIFF_SIZE}-{SLICE_SIZE}.jpg"
    logging.info("Saving produced mosaic to %s", output_filename)
    blended.save(output_filename)

    target_int16 = np.frombuffer(target.tobytes(), dtype=np.uint8).astype(np.int16)
    mosaic_int16 = np.frombuffer(mosaic.tobytes(), dtype=np.uint8).astype(np.int16)
    diff = np.abs(target_int16 - mosaic_int16).astype(np.uint8)
    logging.info(
        "Picture delta: %s per pixel", np.sum(diff) / output_size[0] / output_size[1]
    )
    xored = Image.frombytes(MODE, target.size, diff)
    xored.save(f"output/output-{MODE}-{METRIC}-{DIFF_SIZE}-{SLICE_SIZE}-xor.jpg")
