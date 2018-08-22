import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from PIL import Image
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials
from scipy.spatial.distance import pdist

from lib import Thumb, read_thumbs


def load_media_items():
    with open('media_items.json', 'r') as infile:
        return json.load(infile)


def get_hashes(media_items: List[Dict], size: int, resize: int = None, mode: str = 'L') -> Dict[str, Set[str]]:
    """Computes a map from image hash (in given size) to image ID."""
    hashes = defaultdict(set)
    for m in media_items:
        mid = m['id']
        fname = f'thumbs/{mid}.{size}'
        try:
            im = Image.open(fname).convert(mode)
        except IOError:
            continue
        if im.size != (size, size):
            print(f'Image {mid} has size {im.size} and is_animated: {im.is_animated}')
            continue
        if resize is not None:
            im = im.resize((resize, resize))
        data = im.tobytes()
        if len(data) > 0:
            hash = hashlib.md5()
            hash.update(data)
            digest = hash.digest()
            hashes[digest].add(mid)
    return hashes


def idx_to_ij(idx, N):
    assert N > 0
    i = 0
    while idx >= N - i - 1:
        idx -= N - i - 1
        i += 1
    return i, idx + i + 1


def get_similar_images(thumbs: List[Thumb]) -> List[Tuple[str, str]]:
    print(f'Computing pairwise distances among {len(thumbs)} images')
    data = np.array([t.bytes for t in thumbs])
    dist = pdist(data, 'correlation')

    N = 200
    print(f'Picking the {N} smallest pairs')
    min_idx = np.argpartition(dist, N)[:N]
    return [(thumbs[i].uid, thumbs[j].uid) for i, j in [idx_to_ij(v, len(thumbs)) for v in min_idx[:N]]]


def get_dupes_info(session: AuthorizedSession, dupes: List[Set[str]]) -> List[List[Dict[str, Any]]]:
    out = []
    for dupe in dupes:
        cur = []
        for mid in dupe:
            info = session.get(f'https://photoslibrary.googleapis.com/v1/mediaItems/{mid}').json()
            if info.get('error', {}).get('code') == 404:
                # Delete all versions of that image, in any size
                for filename in os.listdir('thumbs/'):
                    if filename.startswith(mid):
                        print(f'Deleting {filename}')
                        os.unlink(f'thumbs/{filename}')
            else:
                cur.append(info)
        if len(cur) > 1:
            out.append(cur)
    return out


def main():
    media_items = load_media_items()
    ORIG_SIZE = 64
    COMPARE_SIZE = 16

    # First, look for dupes based on image content hashes themselves. Exact matches only.
    hashes = get_hashes(media_items, size=ORIG_SIZE, resize=COMPARE_SIZE)
    dupes = [ids for ids in hashes.values() if len(ids) > 1]
    if dupes:
        print('Found', sum(map(len, dupes)) - len(dupes), 'likely dupes based on hashes')
    else:
        print('No hash-based dupes found, trying dist-based dupes')
        print('Reading thumbnails')
        thumbs = read_thumbs(ORIG_SIZE, 'L', resize=COMPARE_SIZE)
        print(f'Read {len(thumbs)} thumbnails')
        dupes = get_similar_images(thumbs)

    credentials = Credentials.from_authorized_user_file('photos-creds2.json')
    session = AuthorizedSession(credentials)
    dupes_info = get_dupes_info(session, dupes)
    with open('dupe_items.json', 'w') as outfile:
        json.dump(dupes_info, outfile)

    for i, d in enumerate(dupes_info):
        if len(d) != 2:
            print([a['productUrl'] for a in d])
            continue
        a = d[0]
        b = d[1]

        try:
            a_base = a['filename'].split('.')[0].lower()
            b_base = b['filename'].split('.')[0].lower()
            if a_base + '_orig' == b_base:
                print(i, a['filename'], 'is modified of', b['filename'], a['productUrl'])
            elif b_base + '_orig' == a_base:
                print(i, b['filename'], 'is modified of', a['filename'], b['productUrl'])
            elif b_base in a_base and a_base not in b_base:
                print(i, a['filename'], 'contains', b['filename'], a['productUrl'])
            elif a_base in b_base and b_base not in a_base:
                print(i, b['filename'], 'contains', a['filename'], b['productUrl'])
            elif a['filename'].lower() == b['filename'].lower():
                subprocess.call(['/usr/bin/open', a['productUrl']])
                subprocess.call(['/usr/bin/open', b['productUrl']])
            elif re.match(r'.*-.*-.*-.*.JPG', a['filename']):
                print(i, a['filename'], 'ruins', b['filename'], a['productUrl'])
            elif re.match(r'.*-.*-.*-.*.JPG', b['filename']):
                print(i, b['filename'], 'ruins', a['filename'], b['productUrl'])
            else:
                print(i, 'unknown')
        except Exception as e:
            print(a, b, e)


if __name__ == "__main__":
    main()
