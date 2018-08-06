import json
from multiprocessing import Pool
import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from typing import Dict, Tuple
import os


def slice_target(im: Image) -> Dict[Tuple[int, int], bytes]:
    slices = {}
    (w, h) = im.size
    for i in range(0, w, 36):
        for j in range(0, h, 36):
            slices[(i, j)] = np.array([b for b in im.crop((i, j, i + 36, j + 36)).tobytes()], dtype=np.uint8)
    return slices


def read_thumbs() -> Dict[str, bytes]:
    thumbs = {}
    for file in os.listdir('thumbs'):
        try:
            f = Image.open(f'thumbs/{file}').convert('L')
            thumbs[file] = np.array([b for b in f.tobytes()], dtype=np.uint8)
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


if __name__ == "__old_main__":
    print('Reading and slicing target')
    target = Image.open('IMG_3178.JPG')
    print(f'Produced {len(slices)} slices')

    print('Reading thumbnails')
    thumbs = read_thumbs()
    print(f'Read {len(thumbs)} thumbnails')

    output = Image.new('RGB', target.size)

    print('Starting match process...')


    def handle_match(item):
        pos, data = item
        return pos, match(data, thumbs)


    p = Pool(4)
    matches = p.map(handle_match, [x for i, x in enumerate(slice_target(target).items()) if i < 20])
    for ((x, y), match_name) in matches:
        match_data = thumbs[match_name]
        output.paste(Image.frombytes('RGB', (36, 36), match_data), (x, y, x + 36, y + 36))
        print(x, y, match_name)
    output.save('output/output.jpg')

if __name__ == "__oldmain2__":
    print('Reading and slicing target')
    target = Image.open('IMG_3178.JPG').convert('L')
    print(f'Produced {len(slices)} slices')

    print('Reading thumbnails')
    thumbs = read_thumbs()
    print(f'Read {len(thumbs)} thumbnails')

    filter_thumbs = [(k, v) for (k, v) in thumbs.items() if len(v) == 36 * 36 * 1]
    filter_data = np.array([v for (k, v) in filter_thumbs], dtype=np.uint8).astype(np.int32, copy=False)

    output = Image.new('L', target.size)
    for ((x, y), array) in slice_target(target).items():
        diff = np.square(filter_data - array)
        sums = diff.sum(axis=1)
        index = np.argmin(sums)
        match = filter_thumbs[index][0]
        print(x, y, match)
        output.paste(Image.frombytes('L', (36, 36), thumbs[match]), (x, y, x + 36, y + 36))
        if y == 0:
            output.save(f'output/output-L-{x}.jpg')
    output.save(f'output/output-L.jpg')

if __name__ == "__main3__":
    print('Reading and slicing target')
    target = Image.open('IMG_3178.JPG').convert('L')
    slices = slice_target(target)
    print(f'Produced {len(slices)} slices')
    slice_list = list(slices.items())
    slice_data = np.array([array for (pos, array) in slice_list], dtype=np.uint8)

    print('Reading thumbnails')
    thumbs = read_thumbs()
    print(f'Read {len(thumbs)} thumbnails')
    filter_thumbs = [(k, v) for (k, v) in thumbs.items() if len(v) == 36 * 36 * 1]
    filter_data = np.array([v for (k, v) in filter_thumbs], dtype=np.uint8).astype(np.int32, copy=False)

    if not os.path.exists('dist.npy'):
        print('Computing distances')
        dist = cdist(slice_data, filter_data, 'sqeuclidean')
        print('Computed distances', dist.shape)
        np.save('dist.npy', dist)
    else:
        dist = np.load('dist.npy')
    matches = np.argmin(dist, axis=1)

    output = Image.new('L', target.size)
    for i, index in enumerate(matches):
        (x, y) = slice_list[i][0]
        match = filter_thumbs[index][0]
        print(x, y, match)
        output.paste(Image.frombytes('L', (36, 36), thumbs[match]), (x, y, x + 36, y + 36))
    output.save(f'output/output-L.jpg')

if __name__ == "__main__":
    print('Loading costs')
    dist = np.load('dist.npy').transpose()
    print('Loaded cost matrix of size', dist.shape)

    print('Reshaping dist')
    shape = dist.shape
    dist = np.reshape(dist, (dist.size, 1))
    print(dist.shape)

    print('Sorting dist')
    a = np.argsort(dist, axis=0)
    print('Unraveling dist')
    a_idx = np.array(np.unravel_index(a, shape)).T[0]

    print('Matching')
    count = 0
    if not os.path.exists('matches.out'):
        matches = {}
        used = set()
        for (i, j) in a_idx:
            count += 1
            if count % 100000 == 0:
                print(count, len(matches), len(used))
            i = int(i)
            j = int(j)
            if j not in matches and i not in used:
                matches[j] = i
                used.add(i)
            if len(matches) == shape[1]:
                break
        with open('matches.out', 'w') as out:
            json.dump(matches, out)
    else:
        with open('matches.out', 'r') as infile:
            matches = json.load(infile)

    print('Reading and slicing target')
    target = Image.open('IMG_3178.JPG').convert('L')
    slices = slice_target(target)
    slice_list = list(slices.items())
    print(f'Produced {len(slices)} slices')

    print('Reading thumbnails')
    thumbs = read_thumbs()
    print(f'Read {len(thumbs)} thumbnails')

    filter_thumbs = [(k, v) for (k, v) in thumbs.items() if len(v) == 36 * 36 * 1]
    filter_data = np.array([v for (k, v) in filter_thumbs], dtype=np.uint8).astype(np.int32, copy=False)
    output = Image.new('L', target.size)
    for i, j in matches.items():
        (x, y) = slice_list[i][0]
        match = filter_thumbs[j][0]
        output.paste(Image.frombytes('L', (36, 36), thumbs[match]), (x, y, x + 36, y + 36))
    output.save(f'output/output-L.jpg')
