import json
import os
import queue
import time
from pathlib import Path
from threading import Thread

from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from requests import HTTPError


def get_session() -> AuthorizedSession:
    flow = InstalledAppFlow.from_client_secrets_file(
        "app-creds.json",
        scopes=[
            "https://www.googleapis.com/auth/photoslibrary.readonly",
            "https://www.googleapis.com/auth/photoslibrary.sharing",
        ],
        redirect_uri="urn:ietf:wg:oauth:2.0:oob",
    )

    creds = flow.run_local_server()

    creds_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }

    if True:
        del creds_data["token"]
        saved_file = f"photos-creds-{time.time()}.json"
        with open(saved_file, "w") as out:
            json.dump(creds_data, out)
        print(f"Next time use: {saved_file}")

    return flow.authorized_session()


def _actually_list_media_items(session: AuthorizedSession):
    ret = []
    params = {
        "fields": "mediaItems(id,baseUrl,filename,mimeType,productUrl),nextPageToken",
    }
    search_json = {
        "pageSize": 100,
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
            },
            "mediaTypeFilter": {
                "mediaTypes": [
                    "PHOTO",
                ],
            },
        },
    }

    while True:
        rsp = session.post(
            "https://photoslibrary.googleapis.com/v1/mediaItems:search",
            params=params,
            json=search_json,
        ).json()
        if "error" in rsp:
            print(rsp)
            if rsp["error"].get("code", 429) == 503:
                continue
            return ret

        cur = [m for m in rsp.get("mediaItems", [])]
        ret += cur
        print(f"{len(cur)} new items, total {len(ret)}")

        pageToken = rsp.get("nextPageToken")
        if pageToken is None:
            break
        params["pageToken"] = pageToken
    return ret


def _download_image(session: AuthorizedSession, queue, size):
    while True:
        m = queue.get()
        if m.get("mimeType", "").startswith("image/"):
            outfile = f"thumbs/{m['id']}.{size}"
            if not os.path.exists(outfile) or os.stat(outfile).st_size == 0:
                try:
                    with open(outfile, "wb") as out:
                        r = session.get(m["baseUrl"] + f"=w{size}-h{size}-c")
                        r.raise_for_status()
                        out.write(r.content)
                except HTTPError as e:
                    print(e)
                    os.unlink(outfile)
        else:
            print(f"Skipping {m} since {m.get('mimeType','')} is not 'image/'")
        queue.task_done()


def download_images(session, media_items, size):
    concurrent = 30
    download_queue = queue.Queue(concurrent * 2)
    thumbs = Path("thumbs")
    thumbs.mkdir(exist_ok=True)
    for i in range(concurrent):
        t = Thread(target=lambda: _download_image(session, download_queue, size))
        t.daemon = True
        t.start()
    used_ids = set()
    skip_count = 0
    for m in media_items:
        if m["id"] not in used_ids:
            download_queue.put(m)
            used_ids.add(m["id"])
        else:
            skip_count += 1
    download_queue.join()
    print(f"Downloaded {len(used_ids)} images ({skip_count} skipped)")


def list_media_items(session):
    if os.path.exists("media_items.json"):
        with open("media_items.json", "r") as infile:
            return json.load(infile)
    else:
        media_items = _actually_list_media_items(session)
        with open("media_items.json", "w") as out:
            json.dump(media_items, out)
        print("Wrote", len(media_items), "media items out")
        return media_items


def delete_unknown_files(media_items):
    known_ids = set(m["id"] for m in media_items)
    to_delete = set(
        f"thumbs/{f}"
        for f in os.listdir("thumbs")
        if os.path.splitext(f)[0] not in known_ids
    )
    print(len(to_delete), "to delete")
    for f in to_delete:
        os.unlink(f)


if __name__ == "__main__":
    credentials = Credentials.from_authorized_user_file(
        "photos-creds-1637217451.418922.json"
    )
    session = AuthorizedSession(credentials)
    # session = get_session()
    media_items = list_media_items(session)
    print("Read", len(media_items), "media items")
    size = 128
    download_images(session, media_items, size=size)
    delete_unknown_files(media_items)
