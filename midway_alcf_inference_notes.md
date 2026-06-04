# Running Decrypto against the ALCF inference gateway (Polaris/Sophia)

This explains how to run Decrypto against models **hosted by ALCF** through the
[ALCF inference endpoints](https://docs.alcf.anl.gov/services/inference-endpoints/)
gateway, instead of serving a model yourself with vLLM on Midway (see
[`midway_notes.md`](midway_notes.md) for that path).

The gateway is an OpenAI-compatible HTTPS service. That has two nice
consequences:

- **No local GPU / vLLM server is needed.** You don't submit a server slurm job;
  you just run the experiment driver and it makes HTTPS calls.
- **It runs from anywhere with internet + a Globus token** — a Midway login node,
  an ALCF login node, or your laptop. The model runs on ALCF hardware; only the
  game loop runs locally.

Authentication is a Globus access token passed as a Bearer token. We vendored
the official ALCF helper at
[`src/utils/inference_auth_token.py`](src/utils/inference_auth_token.py) and wired
it into the client.

---

## How it plugs into the existing code

A `LocalModel` already builds an `OpenAI(base_url=...)` client and routes through
the generic "vLLM" chat-completion branch in
[`src/agents/role_client.py`](src/agents/role_client.py). The only gateway-specific
needs are (1) the gateway URL and (2) a real token instead of `"dummy_key"`. So:

- `src/types.py` — `LocalModel` gained a `use_globus_auth: bool` field.
- `src/agents/role_client.py` — when `use_globus_auth` is set, `initialize_client`
  calls `get_access_token()` and uses it as the OpenAI `api_key` (Bearer token).
  The import is lazy, so runs that never touch ALCF don't need `globus_sdk`.
- `config/examples/argonne_polaris.yaml` — a self-play smoke-test config pointed
  at the Sophia vLLM endpoint with `use_globus_auth: true`.

Gateway base URLs (from the ALCF docs):

| Cluster | Base URL | Backend |
|---|---|---|
| Sophia | `https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1` | vLLM (full OpenAI surface: chat, completions, embeddings, batch) |
| Metis  | `https://inference-api.alcf.anl.gov/resource_server/metis/api/v1`   | SambaNova (chat completions only) |

We default to **Sophia/vLLM** because it matches the OpenAI chat-completions code
path the runner already uses.

---

## Prerequisite: access to the inference service

Authenticating proves *who* you are; it does **not** grant access. The gateway
authorizes per Globus Group membership tied to an active ALCF allocation. If
your identity isn't in the right group you'll authenticate fine but every call
returns HTTP 401:

```
{"error": {"code": "unauthorized", "message": "Error: Permission denied. User
 (<you>) not part of the Globus Groups applied for University of Chicago."}}
```

To get access: request a project/allocation (e.g. ALCF Director's Discretionary)
that includes the inference service, join the project, and ask ALCF Support to
add your Globus identity to the inference Globus Group. See
<https://docs.alcf.anl.gov/services/inference-endpoints/> and
<https://www.alcf.anl.gov/support-center/get-started/request-allocation>.

If you *do* have access but authenticated as the wrong linked identity, log out
at <https://app.globus.org/logout> and re-run authentication with `--force`,
selecting your ALCF-linked identity.

## One-time setup

1. **`globus_sdk` is installed** in the `vllm-probe` env (4.7.0). If you ever
   recreate the env, add it back:

   ```bash
   pip install "globus-sdk>=3.30"
   ```

2. **Authenticate to Globus** (one time; mints + caches refresh/access tokens
   under `~/.globus/app/.../inference_app/tokens.json`). The stock flow is
   **interactive** — it prints a Globus URL to open in a browser, then reads the
   resulting code from stdin:

   ```bash
   python -m src.utils.inference_auth_token authenticate
   ```

   - Run this in a **real interactive terminal**. Claude Code's `!` runner (and
     other non-TTY contexts) have no interactive stdin, so the stock flow dies
     with `EOFError: EOF when reading a line`.
   - For non-interactive contexts, use the file-based waiter we added
     ([`src/utils/globus_login.py`](src/utils/globus_login.py)): it prints the
     URL, then polls `~/.globus_auth_code.txt` for the code instead of reading
     stdin. Start it (e.g. in the background), open the printed URL, then deliver
     the code with `echo 'CODE' > ~/.globus_auth_code.txt`.
   - Alternative: run `authenticate` on your laptop, then copy the resulting
     `~/.globus/app/.../inference_app/tokens.json` to the same path in your
     cluster home.
   - Access tokens last ~48h and auto-refresh from the cached refresh token on
     every `get_access_token()` call, so you only authenticate once (until the
     refresh token expires from long inactivity).

   Sanity-check the token afterwards:

   ```bash
   python -m src.utils.inference_auth_token get_time_until_token_expiration --units hours
   ```

3. **List the available models** and confirm which are *running* (a stopped model
   cold-starts on first request, which can take minutes):

   ```bash
   python -m src.utils.list_argonne_endpoints
   ```

   (Equivalent raw call:
   `curl -H "Authorization: Bearer $(python -m src.utils.inference_auth_token get_access_token)" https://inference-api.alcf.anl.gov/resource_server/list-endpoints`)

---

## Running the smoke test

1. **Point the config at a model the gateway serves.** Open
   `config/examples/argonne_polaris.yaml` and set `models[0].model_id` to an
   exact name from `list-endpoints` (the default is
   `meta-llama/Meta-Llama-3.1-70B-Instruct`; smaller/cheaper options like
   `meta-llama/Meta-Llama-3.1-8B-Instruct` or `Qwen/Qwen3-32B` also work). The
   `model_key` (`alcf_llama3.1_70B`) is just a label and is referenced by
   `fixed_interceptor` in the same file — keep them consistent if you rename it.

2. **Run the driver** from the project root (no slurm, no server job):

   ```bash
   cd /project/rcc/mehta5/decrypto
   python run.py --config-name=argonne_polaris
   ```

3. **Collect results.** One self-play episode writes
   `results/argonne_smoke/experiment_summary.csv` plus per-episode histories
   under `results/argonne_smoke/`. With `verbose: true` you'll see each turn's
   code/hints/guesses printed live.

---

## Notes / caveats

- **Thinking models.** The config sets `max_reasoning_tokens: 0`, which uses the
  simple one-shot chat path. The two-stage `<think>` budget path (used for
  Qwen3-style reasoning models) relies on `extra_body.continue_final_message`,
  which the gateway may not support — leave it at 0 unless you've confirmed the
  gateway honors prefill/continue.
- **Cold start.** The first request to a model that isn't currently running can
  block for minutes while the gateway spins it up. The client's retry loop
  tolerates timeouts/5xx with backoff, so it should recover.
- **Parallel combos.** `run_experiments` runs each matchup in its own process via
  `ProcessPoolExecutor`; each process fetches the token independently from the
  shared cache. For a one-model smoke test that's a single process. For large
  sweeps, many processes refreshing the token at once could race on the token
  file — fine in practice because access tokens last long enough to not refresh
  mid-run, but worth knowing.
- **No API-cost gate.** `LocalModel` is exempt from the `confirm_include_api_models`
  check (that gate is only for paid `APIModel`s). ALCF usage draws on your
  allocation, not a billed API key.
- **Cloudflare blocks the default Python User-Agent.** The gateway is behind
  Cloudflare, which edge-bans `Python-urllib/x.y` (HTTP 403, body `error code:
  1010`) before the request reaches the backend. `list_argonne_endpoints.py`
  therefore sends a `curl`-like `User-Agent`. The `openai` client (httpx) is not
  affected. If you write your own raw HTTP probe, set a non-default User-Agent.
