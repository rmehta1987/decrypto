"""List the models/endpoints currently served by the ALCF inference gateway.

Use this to discover which `model_id` values are available (and which are
"running" vs cold) before pointing a Decrypto config at the gateway.

    python -m src.utils.list_argonne_endpoints

Requires a prior `python -m src.utils.inference_auth_token authenticate`.
"""

import json
import urllib.request

from src.utils.inference_auth_token import get_access_token

LIST_ENDPOINTS_URL = (
    "https://inference-api.alcf.anl.gov/resource_server/list-endpoints"
)


def list_endpoints():
    token = get_access_token()
    req = urllib.request.Request(
        LIST_ENDPOINTS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            # The gateway sits behind Cloudflare, which edge-bans the default
            # "Python-urllib/x.y" User-Agent (HTTP 403, "error code: 1010").
            # Send a curl-like UA so the request reaches the backend.
            "User-Agent": "curl/8.4.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


if __name__ == "__main__":
    print(json.dumps(list_endpoints(), indent=2))
