import hashlib
import json
import os
from collections import defaultdict

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials
import subprocess

SIZE=16


def load_media_items():
    with open('media_items.json', 'r') as infile:
        return json.load(infile)


def get_hashes(media_items):
    hashes = defaultdict(set)
    for m in media_items:
        mid = m['id']
        fname = f'thumbs/{mid}.{SIZE}'
        if not os.path.exists(fname):
            continue
        with open(fname, 'rb') as infile:
            data = infile.read()
            if len(data) > 0:
                hash = hashlib.md5()
                hash.update(data)
                digest = hash.digest()
                hashes[digest].add(mid)
    return hashes


def get_dupes_info(session, dupes):
    out = []
    for dupe in dupes:
        cur = []
        for mid in dupe:
            info = session.get(f'https://photoslibrary.googleapis.com/v1/mediaItems/{mid}').json()
            if info.get('error', {}).get('code') == 404:
                os.unlink(f'thumbs/{mid}.{SIZE}')
            else:
                cur.append(info)
        if len(cur) > 1:
            out.append(cur)
        if len(out) >= 50:
            break
    return out


if __name__ == "__main__":
    media_items = load_media_items()
    hashes = get_hashes(media_items)
    dupes = [ids for ids in hashes.values() if len(ids) > 1]
    print('Found', sum(map(len, dupes)) - len(dupes), 'likely dupes')

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

        a_base = a['filename'].split('.')[0]
        b_base = b['filename'].split('.')[0]
        if b_base in a_base and a_base not in b_base:
            print(i, a['filename'], 'contains', b['filename'], a['productUrl'])
        elif a_base in b_base and b_base not in a_base:
            print(i, b['filename'], 'contains', a['filename'], b['productUrl'])
        elif a['filename'] == b['filename'] and a['mediaMetadata'] == b['mediaMetadata']:
            subprocess.call(['/usr/bin/open', a['productUrl']])
            subprocess.call(['/usr/bin/open', b['productUrl']])
        else:
            print(i, 'unknown')
