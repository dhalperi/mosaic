import json
import os
from threading import Thread

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

SIZE = 36


def get_credentials():
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
    print(json.dumps(flow.credentials))


def list_media_items(session):
    ret = []
    params = {
        'pageSize': 500,
        'fields': 'mediaItems(id,baseUrl,mimeType),nextPageToken',
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

        download_images(session, cur)

        pageToken = rsp.get('nextPageToken')
        if pageToken is None:
            break
        params['pageToken'] = pageToken

    return ret


def _download_image(session, queue):
    while True:
        m = queue.get()
        if m.get('mimeType', '').startswith('image/'):
            outfile = f"thumbs/{m['id']}.{SIZE}"
            if not os.path.exists(outfile) or os.stat(outfile).st_size == 0:
                with open(outfile, 'wb') as out:
                    r = session.get(m['baseUrl'] + f'=w{SIZE}-h{SIZE}-c')
                    r.raise_for_status()
                    out.write(r.content)
        queue.task_done()


def download_media_items(session):
    media_items = list_media_items(session)
    with open('media_items.json', 'w') as out:
        json.dump(media_items, out)
    print('Wrote', len(media_items), 'media items out')
    return media_items


def download_images(session, media_items):
    import queue
    concurrent = 50
    queue = queue.Queue(concurrent * 2)
    for i in range(16):
        t = Thread(target=lambda: _download_image(session, queue))
        t.daemon = True
        t.start()
    for m in media_items:
        queue.put(m)
    queue.join()


def delete_unknown_files(media_items):
    known_files = set(f"thumbs/{m['id']}.{SIZE}" for m in media_items)
    files = set(f'thumbs/{f}' for f in os.listdir('thumbs'))
    print(len(files - known_files), 'to delete')
    for f in files - known_files:
        os.unlink(f)


if __name__ == "__main__":
    credentials = Credentials.from_authorized_user_file('photos-creds2.json')
    session = AuthorizedSession(credentials)
    media_items = download_media_items(session)
    print('Read', len(media_items), 'media items')

    delete_unknown_files(media_items)
