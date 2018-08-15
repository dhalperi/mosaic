import json
import os
import queue
from threading import Thread

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from requests import HTTPError

SIZE = 36


def get_session():
    flow = Flow.from_client_secrets_file(
        'app-creds.json',
        scopes=['https://www.googleapis.com/auth/photoslibrary.readonly',
                'https://www.googleapis.com/auth/photoslibrary.sharing'],
        redirect_uri='urn:ietf:wg:oauth:2.0:oob')

    auth_url, _ = flow.authorization_url(prompt='consent')

    print('Please go to this URL: {}'.format(auth_url))
    code = input('Enter the authorization code: ')
    token = flow.fetch_token(code=code)
    print(json.dumps(token))
    # print(json.dumps(flow.credentials))
    return flow.authorized_session()


def _actually_list_media_items(session):
    ret = []
    params = {
        'pageSize': 500,
        'fields': 'mediaItems(id,baseUrl,filename,mimeType,productUrl),nextPageToken',
    }
    search_json = {
        "filters": {
            "includeArchivedMedia": False,
            "contentFilter": {
                "excludedContentCategories": [
                    "DOCUMENTS",
                    "RECEIPTS",
                    "SCREENSHOTS",
                    "UTILITY",
                    "WHITEBOARDS",
                ]
            }
        },
    }

    while True:
        rsp = session.post(
            'https://photoslibrary.googleapis.com/v1/mediaItems:search',
            params=params, json=search_json,
        ).json()

        cur = [m for m in rsp.get('mediaItems', [])]
        ret += cur
        print(f'{len(cur)} new items, total {len(ret)}')

        pageToken = rsp.get('nextPageToken')
        if pageToken is None:
            break
        params['pageToken'] = pageToken
    return ret


def _download_image(session, queue, size):
    while True:
        m = queue.get()
        if m.get('mimeType', '').startswith('image/'):
            outfile = f"thumbs/{m['id']}.{size}"
            if not os.path.exists(outfile) or os.stat(outfile).st_size == 0:
                try:
                    with open(outfile, 'wb') as out:
                        r = session.get(m['baseUrl'] + f'=w{size}-h{size}-c')
                        r.raise_for_status()
                        out.write(r.content)
                except HTTPError as e:
                    print(e)
                    os.unlink(outfile)
        queue.task_done()


def download_images(session, media_items, size=SIZE):
    concurrent = 30
    download_queue = queue.Queue(concurrent * 2)
    for i in range(16):
        t = Thread(target=lambda: _download_image(session, download_queue, size))
        t.daemon = True
        t.start()
    used_ids = set()
    for m in media_items:
        if m['id'] not in used_ids:
            download_queue.put(m)
            used_ids.add(m['id'])
    download_queue.join()


def list_media_items(session):
    if os.path.exists('media_items.json'):
        with open('media_items.json', 'r') as infile:
            return json.load(infile)
    else:
        media_items = _actually_list_media_items(session)
        with open('media_items.json', 'w') as out:
            json.dump(media_items, out)
        print('Wrote', len(media_items), 'media items out')
        return media_items


def delete_unknown_files(media_items):
    known_ids = set(m['id'] for m in media_items)
    to_delete = set(f'thumbs/{f}' for f in os.listdir('thumbs') if os.path.splitext(f)[0] not in known_ids)
    print(len(to_delete), 'to delete')
    for f in to_delete:
        os.unlink(f)


if __name__ == "__main__":
    credentials = Credentials.from_authorized_user_file('photos-creds2.json')
    session = AuthorizedSession(credentials)
    # session = get_session()
    media_items = list_media_items(session)
    print('Read', len(media_items), 'media items')
    size = 36
    download_images(session, media_items, size=size)
    delete_unknown_files(media_items)
