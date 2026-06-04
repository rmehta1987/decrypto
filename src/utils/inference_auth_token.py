"""ALCF inference-endpoints authentication helper.

Vendored verbatim from
https://github.com/argonne-lcf/inference-endpoints/blob/main/inference_auth_token.py

Mints and refreshes a Globus access token for the ALCF inference gateway
(https://inference-api.alcf.anl.gov). Decrypto imports `get_access_token()` to
use as the Bearer token / OpenAI `api_key` when talking to the gateway (see
`src/agents/role_client.py`).

One-time interactive login (opens a Globus auth URL, stores tokens under
`~/.globus/app/<client_id>/inference_app/tokens.json`):

    python -m src.utils.inference_auth_token authenticate

Tokens auto-refresh on subsequent calls via the stored refresh token.
"""

import globus_sdk
from globus_sdk.login_flows import LocalServerLoginFlowManager
import os.path
import time

APP_NAME = "inference_app"
AUTH_CLIENT_ID = "58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944"
GATEWAY_CLIENT_ID = "681c10cc-f684-4540-bcd7-0b4df3bc26ef"
GATEWAY_SCOPE = f"https://auth.globus.org/scopes/{GATEWAY_CLIENT_ID}/action_all"
TOKENS_PATH = f"{os.path.expanduser('~')}/.globus/app/{AUTH_CLIENT_ID}/{APP_NAME}/tokens.json"

GA_PARAMS = globus_sdk.gare.GlobusAuthorizationParameters(
    session_required_policies=["83732ff2-9c42-4548-b5ce-17e498c84f6a"]
)

class DomainBasedErrorHandler:
    def __call__(self, app, error):
        print(f"Encountered error '{error}', initiating login...")
        app.login(auth_params=GA_PARAMS)

def get_auth_object(force=False):
    app = globus_sdk.UserApp(
        APP_NAME,
        client_id=AUTH_CLIENT_ID,
        scope_requirements={GATEWAY_CLIENT_ID: [GATEWAY_SCOPE]},
        config=globus_sdk.GlobusAppConfig(
            request_refresh_tokens=True,
            token_validation_error_handler=DomainBasedErrorHandler()
        ),
    )
    if force:
        app.login(auth_params=GA_PARAMS)
    auth = app.get_authorizer(GATEWAY_CLIENT_ID)
    return auth

def get_access_token():
    auth = get_auth_object(force=False)
    auth.ensure_valid_token()
    return auth.access_token

def get_time_until_token_expiration(units="seconds"):
    auth = get_auth_object(force=False)
    now = time.time()
    delta_t = auth.expires_at - now
    if units == "seconds":
        delta_t = delta_t
    elif units == "minutes":
        delta_t = delta_t / 60
    elif units == "hours":
        delta_t = delta_t / 3600
    else:
        return "Error: units must be 'seconds', 'minutes', or 'hours'."
    return round(delta_t, 2)

if __name__ == "__main__":
    import argparse
    class InferenceAuthError(Exception):
        pass
    AUTHENTICATE_ACTION = "authenticate"
    GET_ACCESS_TOKEN_ACTION = "get_access_token"
    GET_TOKEN_EXPIRATION_ACTION = "get_time_until_token_expiration"
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=[AUTHENTICATE_ACTION,
                        GET_ACCESS_TOKEN_ACTION, GET_TOKEN_EXPIRATION_ACTION])
    parser.add_argument("--units", choices=['seconds', 'minutes', 'hours'],
                        default='seconds', help="Units for token expiration time")
    parser.add_argument("-f", "--force", action="store_true",
                        help="authenticate from scratch")
    args = parser.parse_args()
    if args.action == AUTHENTICATE_ACTION:
        _ = get_auth_object(force=True)
    elif args.action == GET_ACCESS_TOKEN_ACTION:
        if not os.path.isfile(TOKENS_PATH):
            raise InferenceAuthError('Token missing. Run authenticate action first.')
        if args.force:
            raise InferenceAuthError(f"Cannot use --force with {GET_ACCESS_TOKEN_ACTION}.")
        print(get_access_token())
    elif args.action == GET_TOKEN_EXPIRATION_ACTION:
        if not os.path.isfile(TOKENS_PATH):
            raise InferenceAuthError('Token missing. Run authenticate action first.')
        print(get_time_until_token_expiration(args.units))
