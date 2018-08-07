import json
import os

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

SIZE = 16


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
    while True:
        print('Current # of media items is', len(ret))
        rsp = session.get(
            'https://photoslibrary.googleapis.com/v1/mediaItems',
            params=params,
        ).json()
        cur = [m for m in rsp.get('mediaItems', [])]
        pageToken = rsp.get('nextPageToken')
        ret += cur
        download_images(session, cur)

        if pageToken is None:
            break
        params['pageToken'] = pageToken

    return ret


def download_media_items(session):
    media_items = list_media_items(session)
    with open('media_items.json', 'w') as out:
        json.dump(media_items, out)
    print('Wrote', len(media_items), 'media items out')
    return media_items


def download_images(session, media_items):
    for i, m in enumerate(media_items):
        if (i + 1) % 100 == 0:
            print(i)
        if m.get('mimeType', '').startswith('image/'):
            outfile = f"thumbs/{m['id']}.{SIZE}"
            if not os.path.exists(outfile):
                with open(outfile, 'wb') as out:
                    r = session.get(m['baseUrl'] + f'=w{SIZE}-h{SIZE}-c')
                    r.raise_for_status()
                    out.write(r.content)


if __name__ == "__main__":
    credentials = Credentials.from_authorized_user_file('photos-creds2.json')
    session = AuthorizedSession(credentials)
    media_items = download_media_items(session)
    print('Read', len(media_items), 'media items')
