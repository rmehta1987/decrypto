#!/usr/bin/env python3
"""
Option-B smoke test: can we reach the Anthropic (Claude) API from *this* host,
through the ALCF egress proxy when one is required?

Background (see polaris_pbs_notes.md and slurm/smoke_polaris.pbs):
  - Polaris LOGIN nodes have direct outbound internet  -> no proxy needed.
  - Polaris COMPUTE nodes have NO direct internet; outbound HTTPS must tunnel
    through http://proxy.alcf.anl.gov:3128. The node sets http_proxy for you, but
    our vLLM job scripts deliberately `unset http_proxy` (in-cluster traffic would
    otherwise hang routing through the air-gapped proxy -- this was the 900s health
    timeout in smoke jid 7186962). So a job that ALSO wants to call Claude must
    re-establish the proxy for the *external* hop while keeping in-cluster hosts in
    no_proxy. This script validates exactly that external hop.

Checks, in order (fails fast with a clear message + a distinct exit code):
  1. anthropic SDK importable (i.e. you are in the decrypto-serve venv).
  2. TCP / HTTP-CONNECT reachability to the API endpoint (through the proxy if set).
  3. ANTHROPIC_API_KEY present (from the environment or a .env file).
  4. A real, minimal client.messages.create() round-trip.
  5. A real `claude -p` headless-CLI round-trip (the path RoleClient.use_cli
     actually uses in production -- see src/agents/role_client.py). This uses
     the CLI's own OAuth session, NOT the API key, so it runs regardless of
     whether step 3/4 have a key: it is the load-bearing check for whether the
     ALCF proxy tunnels the CLI's own traffic, which is a separate question
     from whether it tunnels the raw `anthropic` SDK's traffic in step 4.

Safe to run BEFORE you have a key: steps 1-2 validate the network path today, and
step 3 then reports that the key is the only thing still missing. Step 5 needs
no key at all.

Run (inside the venv):
  python slurm/smoke_anthropic_proxy.py                 # login node (direct egress)
  python slurm/smoke_anthropic_proxy.py --alcf-proxy    # compute node (via ALCF proxy)
  python slurm/smoke_anthropic_proxy.py --proxy http://host:port
  python slurm/smoke_anthropic_proxy.py --model claude-sonnet-4-6
  python slurm/smoke_anthropic_proxy.py --skip-cli-check # SDK/key path only

Exit codes: 0 ok | 2 network/proxy unreachable | 3 no API key | 4 API call failed |
            5 env/import error | 6 CLI round-trip failed
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from typing import Optional, Tuple
from urllib.parse import urlsplit

ENDPOINT_HOST = "api.anthropic.com"
ENDPOINT_PORT = 443
ALCF_PROXY = "http://proxy.alcf.anl.gov:3128"

EXIT_OK, EXIT_NET, EXIT_NOKEY, EXIT_API, EXIT_ENV, EXIT_CLI = 0, 2, 3, 4, 5, 6


def mask(secret):  # type: (str) -> str
    if not secret:
        return "(unset)"
    if len(secret) > 14:
        return "%s...%s (len %d)" % (secret[:8], secret[-4:], len(secret))
    return "(set, len %d)" % len(secret)


def resolve_proxy(args):  # type: (argparse.Namespace) -> Optional[str]
    if args.proxy:
        return args.proxy
    if args.alcf_proxy:
        return ALCF_PROXY
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        if os.environ.get(var):
            return os.environ[var]
    return None


def no_proxy_shadows_endpoint():  # type: () -> Optional[str]
    """Return the no_proxy entry that would (wrongly) bypass the proxy for the
    Anthropic endpoint, or None. Mirrors httpx/requests suffix-match semantics."""
    raw = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    for entry in (x.strip() for x in raw.split(",")):
        if not entry:
            continue
        if entry == "*":
            return entry
        dom = entry.lstrip(".")
        if ENDPOINT_HOST == dom or ENDPOINT_HOST.endswith("." + dom):
            return entry
    return None


def probe_network(proxy, timeout):  # type: (Optional[str], float) -> Tuple[bool, str]
    """Validate we can actually reach the API endpoint from this host.

    With a proxy, issue a real HTTP CONNECT and confirm the proxy will tunnel to
    api.anthropic.com:443 -- this isolates "proxy reachable + allows egress" from
    "API key valid". Without a proxy, a direct TCP connect (login-node path)."""
    if proxy:
        parts = urlsplit(proxy)
        phost, pport = parts.hostname, (parts.port or 3128)
        if not phost:
            return False, "could not parse proxy URL: %r" % proxy
        try:
            sock = socket.create_connection((phost, pport), timeout=timeout)
        except OSError as exc:
            return False, "cannot reach proxy %s:%s: %s" % (phost, pport, exc)
        try:
            req = (
                "CONNECT %s:%d HTTP/1.1\r\nHost: %s:%d\r\n\r\n"
                % (ENDPOINT_HOST, ENDPOINT_PORT, ENDPOINT_HOST, ENDPOINT_PORT)
            )
            sock.sendall(req.encode())
            raw = sock.recv(4096).decode("latin-1", "replace").splitlines()
            line = raw[0] if raw else ""
        except OSError as exc:
            return False, "proxy CONNECT to %s failed: %s" % (ENDPOINT_HOST, exc)
        finally:
            sock.close()
        if " 200 " in (" " + line + " "):
            return True, "proxy %s:%s tunnels to %s:443 (%s)" % (
                phost, pport, ENDPOINT_HOST, line)
        return False, "proxy refused CONNECT to %s: %r" % (ENDPOINT_HOST, line)
    # No proxy -> expect direct egress (login node).
    try:
        socket.create_connection((ENDPOINT_HOST, ENDPOINT_PORT), timeout=timeout).close()
    except OSError as exc:
        return False, (
            "no proxy set and direct TCP to %s:443 failed: %s\n"
            "       on a compute node, re-run with --alcf-proxy"
            % (ENDPOINT_HOST, exc)
        )
    return True, "direct TCP to %s:443 OK (no proxy)" % ENDPOINT_HOST


def check_cli(model, timeout):  # type: (str, float) -> Tuple[bool, str]
    """Round-trip through the ACTUAL path RoleClient.use_cli uses: headless
    `claude -p`, prompt over stdin, --system-prompt full override, --tools ""
    (see src/agents/role_client.py:_claude_cli_completion). Auth is whatever
    OAuth session `claude` already has on this host -- no key involved, so a
    failure here isolates "the CLI's own network stack doesn't honor the
    proxy" from "the API key/SDK path doesn't work" (checked separately above).
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False, "`claude` binary not found on PATH (need ~/.local/bin on PATH in this job)"

    token = "DECRYPTO_CLI_PROXY_OK"
    cmd = [
        claude_bin, "-p",
        "--model", model,
        "--system-prompt", "You are a smoke-test assistant.",
        "--output-format", "json",
        "--tools", "",
        "--no-session-persistence",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input="Reply with exactly this token and nothing else: %s" % token,
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, (
            "claude -p timed out after %ss -- likely the proxy is not tunneling "
            "the CLI's own traffic even though the raw SDK/network checks passed"
            % timeout
        )
    if not proc.stdout:
        return False, "claude -p produced no stdout (rc=%s): %s" % (
            proc.returncode, proc.stderr[:500])
    try:
        data = json.loads(proc.stdout)
    except Exception as exc:
        return False, "claude -p stdout not JSON (%s): %s" % (exc, proc.stdout[:500])
    if data.get("is_error"):
        return False, "claude -p reported is_error: %s" % data.get("result")
    result = data.get("result") or ""
    ok = token in result
    detail = "round-trip %s: %r (model=%s, cost=$%.4f, %sms)" % (
        "OK" if ok else "returned unexpected content",
        result, data.get("modelUsage", {}), data.get("total_cost_usd", 0),
        data.get("duration_ms", "?"),
    )
    return ok, detail


def main():  # type: () -> int
    ap = argparse.ArgumentParser(
        description="Anthropic API reachability smoke test (Option B).")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="model id (default: claude-sonnet-4-6)")
    ap.add_argument("--proxy", help="explicit egress proxy URL (overrides env)")
    ap.add_argument("--alcf-proxy", action="store_true",
                    help="use %s" % ALCF_PROXY)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--skip-network-probe", action="store_true")
    ap.add_argument("--skip-cli-check", action="store_true",
                    help="skip the `claude -p` round-trip (step 5)")
    args = ap.parse_args()

    # Best-effort .env load (matches slurm/test_anthropic_api.py).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    proxy = resolve_proxy(args)
    print("=== Anthropic API smoke (Option B) ===")
    print("host             : %s" % socket.gethostname())
    print("python           : %s (%s)" % (sys.version.split()[0], sys.executable))
    print("endpoint         : https://%s" % ENDPOINT_HOST)
    print("egress proxy     : %s" % (proxy or "(none -- direct)"))
    print("ANTHROPIC_API_KEY: %s" % mask(os.environ.get("ANTHROPIC_API_KEY", "")))
    shadow = no_proxy_shadows_endpoint()
    if proxy and shadow:
        print("WARNING          : no_proxy entry %r would bypass the proxy for %s. "
              "The SDK call below pins the proxy explicitly, but env-based clients "
              "(curl / the openai SDK) would go direct and fail on a compute node."
              % (shadow, ENDPOINT_HOST))

    # 1) SDK import -------------------------------------------------------------
    try:
        import anthropic
        from anthropic import Anthropic, DefaultHttpxClient
    except Exception as exc:
        print("\nFAIL [env]: cannot import anthropic (%s)." % exc)
        print("       Activate the decrypto-serve venv (it has `anthropic`), "
              "or `pip install anthropic`.")
        return EXIT_ENV
    print("anthropic sdk    : %s" % anthropic.__version__)

    # 2) Network reachability ---------------------------------------------------
    if not args.skip_network_probe:
        ok, detail = probe_network(proxy, args.timeout)
        print("\n[net] %s: %s" % ("OK  " if ok else "FAIL", detail))
        if not ok:
            return EXIT_NET

    # 3) Real claude-CLI round-trip (OAuth session, no API key needed) ----------
    #    This is the path RoleClient.use_cli actually exercises in production,
    #    and it is independent of the ANTHROPIC_API_KEY-gated SDK check below.
    cli_exit = EXIT_OK
    if not args.skip_cli_check:
        ok, detail = check_cli(args.model, args.timeout)
        print("\n[cli] %s: %s" % ("OK  " if ok else "FAIL", detail))
        if not ok:
            cli_exit = EXIT_CLI

    # 4) Key present? (gates the SDK round-trip only) ----------------------------
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n[key]: ANTHROPIC_API_KEY not set (environment or .env) -- "
              "skipping the SDK round-trip. The CLI path above does not need it.")
        return cli_exit

    # 4) Real API round-trip (proxy pinned explicitly for deterministic routing)-
    client_kwargs = dict(timeout=args.timeout, max_retries=2)
    if proxy:
        client_kwargs["http_client"] = DefaultHttpxClient(proxy=proxy)
    client = Anthropic(**client_kwargs)

    token = "DECRYPTO_SMOKE_OK"
    try:
        resp = client.messages.create(
            model=args.model,
            max_tokens=args.max_tokens,
            messages=[{
                "role": "user",
                "content": "Reply with exactly this token and nothing else: " + token,
            }],
        )
    except anthropic.AuthenticationError as exc:
        print("\nFAIL [api]: authentication failed (bad/expired key): %s" % exc)
        return EXIT_API
    except anthropic.APIConnectionError as exc:
        print("\nFAIL [api]: connection error (proxy/network) reaching %s: %s"
              % (ENDPOINT_HOST, exc))
        print("       If on a compute node, ensure the proxy is set (--alcf-proxy).")
        return EXIT_API
    except anthropic.APIStatusError as exc:
        print("\nFAIL [api]: HTTP %s %s: %s"
              % (exc.status_code, getattr(exc, "type", ""), exc.message))
        return EXIT_API
    except Exception as exc:
        print("\nFAIL [api]: unexpected error: %s: %s" % (type(exc).__name__, exc))
        return EXIT_API

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    usage = resp.usage
    print("\n[api] OK : round-trip succeeded")
    print("      model      : %s" % resp.model)
    print("      stop_reason: %s" % resp.stop_reason)
    print("      reply      : %r" % text)
    print("      tokens     : in=%s out=%s" % (usage.input_tokens, usage.output_tokens))
    print("      request_id : %s" % getattr(resp, "_request_id", None))
    if token not in text:
        print("      NOTE: reply did not contain %r -- transport works, "
              "content unexpected." % token)
    print("\n=== SMOKE GREEN: Option B (Claude via %s) works from this host ==="
          % ("proxy" if proxy else "direct"))
    return cli_exit


if __name__ == "__main__":
    sys.exit(main())
