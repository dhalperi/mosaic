from google.oauth2.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession
import json
import os

# from oauth2client.client import flow_from_clientsecrets
# import requests


# def get_credentials():
#     flow = flow_from_clientsecrets('credentials.json',
#                                    scope='https://www.googleapis.com/auth/photoslibrary.readonly',
#                                    redirect_uri='urn:ietf:wg:oauth:2.0:oob')
#     auth_uri = flow.step1_get_authorize_url()
#     print('Visit', auth_uri, 'and paste the code here:')
#     access_token = input().strip()
#     return flow.step2_exchange(access_token)


config = {
    'api_endpoint': 'https://photoslibrary.googleapis.com/v1',
}


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
        if pageToken is None:
            print(rsp)
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
        if (i+1) % 100 == 0:
            print(i)
        if m.get('mimeType', '').startswith('image/'):
            outfile = f"thumbs/{m['id']}.36"
            if not os.path.exists(outfile):
                with open(outfile, 'wb') as out:
                    out.write(session.get(m['baseUrl'] + '=w36-h36-c').content)


if __name__ == "__main__":
    credentials = Credentials.from_authorized_user_file('photos-creds.json')
    session = AuthorizedSession(credentials)
    media_items = download_media_items(session)
    print('Read', len(media_items), 'media items')
