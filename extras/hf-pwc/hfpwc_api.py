"""
Client for the public API of Papers with Code (https://paperswithcode.co/docs/api).

Requests are spaced to stay below the limits of 60 (lists) and 120 requests per minute,
and retried on 429 and 5xx errors. The default Python User-Agent gets a 403.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator, Optional

# ==================================================================================================

BASE_URL = "https://paperswithcode.co/api/v1"
USER_AGENT = (
    "OpenCodePapers-import/1.0 (+https://gitlab.com/OpenCodePapers/OpenCodePapers)"
)

LIST_INTERVAL = 1.1
DETAIL_INTERVAL = 0.6
MAX_ATTEMPTS = 8

_last_request = 0.0

# ==================================================================================================


def get(path: str, params: Optional[dict] = None, is_list: bool = False):
    """JSON response of `path` (like "/tasks/"), None if the record doesn't exist."""
    global _last_request  # pylint: disable=global-statement

    url = BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}

    for attempt in range(MAX_ATTEMPTS):
        wait = _last_request + (LIST_INTERVAL if is_list else DETAIL_INTERVAL)
        if time.time() < wait:
            time.sleep(wait - time.time())
        _last_request = time.time()

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code not in (429, 500, 502, 503, 504):
                raise
            retry_after = e.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else 15 * (attempt + 1)
            print(f"HTTP {e.code} for {url}, retrying in {delay}s", flush=True)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as e:
            delay = 15 * (attempt + 1)
            print(f"{e} for {url}, retrying in {delay}s", flush=True)
            time.sleep(delay)

    raise RuntimeError(f"Request failed {MAX_ATTEMPTS} times: {url}")


def get_all(path: str, params: Optional[dict] = None) -> Iterator[dict]:
    """All results of a paginated list endpoint, in stable id order."""
    params = dict(params or {})
    params.setdefault("ordering", "id")
    params["page_size"] = 100
    page = 1
    while page:
        params["page"] = page
        data = get(path, params, is_list=True)
        yield from data["results"]
        page = data.get("next_page")
