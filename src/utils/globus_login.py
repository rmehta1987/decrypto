"""Non-interactive Globus login for the ALCF inference gateway.

The stock `inference_auth_token authenticate` flow reads the authorization code
from an interactive stdin via `input()`. That fails in environments without an
attached TTY (e.g. Claude Code's `!` runner, `nohup`, CI), with:

    EOFError: EOF when reading a line

This module runs the *same* stock flow (so PKCE verifier and token storage are
handled by globus_sdk exactly as usual — everything stays in one process), but
replaces the code prompt with a poll on a file. Workflow:

  1) Start the waiter (prints the Globus auth URL, then blocks):
         python -m src.utils.globus_login
  2) Open the printed URL, log in with your ALCF-linked identity, copy the code.
  3) Deliver the code by writing it to the file (default ~/.globus_auth_code.txt):
         echo 'PASTE_CODE_HERE' > ~/.globus_auth_code.txt

The waiter then exchanges the code and caches tokens at
`inference_auth_token.TOKENS_PATH`, after which `get_access_token()` works
normally. Override the file path with GLOBUS_AUTH_CODE_FILE.
"""

import builtins
import os
import time

from src.utils import inference_auth_token as iat

CODE_FILE = os.environ.get(
    "GLOBUS_AUTH_CODE_FILE", os.path.expanduser("~/.globus_auth_code.txt")
)
TIMEOUT_S = int(os.environ.get("GLOBUS_AUTH_TIMEOUT", "1200"))
POLL_S = 2


def _wait_for_code(prompt=""):
    """Drop-in replacement for input(): block until CODE_FILE has a code."""
    print(f"\n[globus_login] Waiting for the authorization code at: {CODE_FILE}", flush=True)
    print(f"[globus_login] After logging in, run:  echo 'YOUR_CODE' > {CODE_FILE}", flush=True)
    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        if os.path.isfile(CODE_FILE):
            with open(CODE_FILE) as f:
                code = f.read().strip()
            if code:
                os.remove(CODE_FILE)
                print("[globus_login] Got code; exchanging for tokens...", flush=True)
                return code
        time.sleep(POLL_S)
    raise TimeoutError(
        f"[globus_login] No code written to {CODE_FILE} within {TIMEOUT_S}s."
    )


if __name__ == "__main__":
    # Clear any stale code so we don't consume a leftover value.
    if os.path.isfile(CODE_FILE):
        os.remove(CODE_FILE)
    # CommandLineLoginFlowManager calls the builtin input() for the code.
    builtins.input = _wait_for_code
    iat.get_auth_object(force=True)
    print(f"[globus_login] Authentication complete. Tokens cached at: {iat.TOKENS_PATH}", flush=True)
