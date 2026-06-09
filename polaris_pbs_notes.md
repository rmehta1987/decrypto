# Running Decrypto's self-hosted vLLM pipeline on ALCF Polaris (PBS)

Living document. It explains how to stand up and run the **Decrypto self-hosted
vLLM serving pipeline** on ALCF **Polaris** (PBS Pro), and records the changes
needed to port the working RCC **Midway** (Slurm) version. It is the PBS
equivalent of [`midway_notes.md`](midway_notes.md) — read that first for the
Slurm template and the load-bearing fixes.

> **Scope:** self-hosted serving only — submit a job that runs `vllm serve` on a
> Polaris A100 node, then a second job runs the Decrypto game loop against it
> over HTTP. This is **NOT** the ALCF inference-*gateway* path
> (`midway_alcf_inference_notes.md` / `argonne_polaris.yaml` / `inference_auth_token.py`),
> which makes HTTPS calls to models ALCF already hosts and is blocked on Globus.
>
> The proven Polaris cluster facts and the gotcha log come from a sibling project
> on this same allocation: **`../MARSHAL/polaris_pbs_notes.md`** (treat its
> cluster-facts table as ground truth). MARSHAL is a *training* pipeline; only its
> **cluster-level** facts apply to us (we serve/infer, no Ray/DeepSpeed/weight-sync).

If you just want to run the smoke test, jump to [Running a smoke test](#running-a-smoke-test).
If something breaks, the [Decisions / changes log](#decisions--changes-log) at the
bottom captures what we hit and how we fixed it.

---

## ▶ STATUS: <!--STATUS-->GREEN ✅ (2026-06-07) — self-hosted vLLM pipeline proven end-to-end on Polaris<!--/STATUS-->

The Decrypto self-play smoke closed end-to-end on a Polaris A100 (single-job jid
**7186966**, node `x3106c0s13b0n0`): venv staged from tarball (12 s) → `vllm serve`
healthy (21 s) → registered via `DECRYPTO_SERVERS_FILE` with its HSN IP →
`ping_servers` got a healthy reply over `http://10.201.3.1:8159/v1` (0.55 s) →
`run.py` discovered the server and ran **1/1** self-play episodes →
`results/polaris_smoke/experiment_summary.csv` written → vLLM stopped;
`Exit_status=0`. (Gameplay itself is garbage — Qwen2.5-0.5B fails the JSON-format
retries — a model-capacity issue, **not** a pipeline issue, exactly as Midway
documented.)

```bash
# Reproduce (debug queue):
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/decrypto
qsub slurm/probe_vllm_polaris.pbs   # prove the toolchain first
qsub slurm/smoke_polaris.pbs        # server + 1 episode + results, one node
```

---

## Cluster facts (Polaris / ALCF)

Confirmed on this allocation (June 2026); cross-checked against MARSHAL's bring-up.

| Item | Value |
|---|---|
| Scheduler | **PBS Pro** (`qsub`/`qstat`/`qdel`), login host `polaris-login-04` |
| Account (`-A`) | **`lighthouse-uchicago`** ⚠️ NOT "Uchicago-lighthouse" (PBS rejects that). Confirmed via `sbank-list-allocations`: alloc 12374, ~17,175 node-h available |
| Queue (`-q`) | **`debug`** (1–2 nodes, ≤1 h walltime) for the smoke. Also `debug-scaling`, `prod` |
| GPU | **4× NVIDIA A100 40 GB** (HBM2, sm_80) per node. ⚠️ far tighter than Midway's H200 (~140 GB) |
| CPU / RAM | AMD EPYC Milan, 64 hardware threads (`ncpus=64`), 512 GiB |
| Filesystems | `home`, `eagle`, `grand`. Jobs **must** declare `-l filesystems=home:eagle` or PBS rejects them |
| Select line | `-l select=1:ncpus=64:ngpus=4` (do **NOT** add `:system=polaris`) |
| Node-local scratch | `/local/scratch` (RAID0 SSD) — fast + writable; used for `TMPDIR` + venv staging. Falls back to `/tmp` |
| Conda module | `module use /soft/modulefiles && module load conda/2025-09-25 && conda activate base` (build the venv on top) |
| Native CUDA | `/soft/compilers/cudatoolkit/cuda-12.4.1`; driver supports ≥ CUDA 12.8 → cu124 wheels are safe |
| Member base | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5` (repo, venv, models, caches) |
| Project root | `…/members/mehta5/decrypto` |
| Model store | `…/members/mehta5/models/` (`Qwen2.5-0.5B-Instruct` staged) |
| Serving venv | `…/members/mehta5/conda-envs/decrypto-serve` (+ tarball `…/members/mehta5/decrypto-serve-venv.tar`) |
| Caches | `…/members/mehta5/{hf_cache,triton_cache,torchinductor_cache}` |
| Job ids | `1234567.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`; bare tag via `${PBS_JOBID%%.*}` |
| Cross-node (HSN) | compute nodes `x3001c0s7b1n0`; HSN IPs `10.201.x.x`; FQDN `<node>.hsn.cm.polaris.alcf.anl.gov` |

### PBS submit idiom (single GPU node)

```bash
qsub -A lighthouse-uchicago -q debug \
     -l select=1:ncpus=64:ngpus=4 -l filesystems=home:eagle \
     -l walltime=01:00:00 slurm/<script>.pbs
# (these are baked into each script's #PBS header)
```

---

## One-time setup

Done once per account; on a **login node** (has internet + HF reachability).

1. **Build the serving venv + tarball.** The stack mirrors MARSHAL's proven cu124
   set and adds the Decrypto runner deps, so **one env hosts both** the `vllm
   serve` server and the HTTP-only game runner:

   ```
   torch 2.6.0+cu124 · vllm 0.8.4 · transformers 4.51.2 · tokenizers <0.22 · numpy <2
   + runner: litellm anthropic gensim nltk hydra-core openai python-dotenv scipy pandas tqdm requests
   ```

   ```bash
   bash slurm/build_decrypto_serve_venv.sh
   ```

   This builds `…/conda-envs/decrypto-serve` and packs it into
   `…/decrypto-serve-venv.tar`. **Why a tarball?** eagle (Lustre) is fast for big
   sequential reads but catastrophic at the ~70k-tiny-file metadata storm
   `import torch`/vllm triggers on a cold compute node (MARSHAL measured ~19 min /
   effective hang). Every job extracts the **one** big file onto `/local/scratch`
   (~12 s) and runs from there.

2. **Pre-download every model** you plan to serve into the model store
   (compute nodes are **offline**). The smoke model is staged at
   `…/members/mehta5/models/Qwen2.5-0.5B-Instruct`. To add another:

   ```bash
   # on a login node, in the decrypto-serve venv:
   huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
     --local-dir /lus/eagle/projects/lighthouse-uchicago/members/mehta5/models/Qwen2.5-72B-Instruct
   ```

---

## Running a smoke test

> **⚠️ Which path to use — the `debug` queue forces a single job.** `debug`
> enforces **`max_run=1`** and **`queued_jobs_threshold=1`** per user
> (`qstat -Qf debug`). The faithful Midway-style **two-job** pattern (a server job
> + a dependent experiment job) therefore **cannot run on debug**: the two jobs
> can never be Running at once, and because the experiment `qdel`s the server they
> would deadlock. So:
> - **On `debug` (the smoke):** use the **single-job** `slurm/smoke_polaris.pbs`
>   (server + experiment in one job, one node).
> - **On a queue allowing ≥2 concurrent jobs** (prod / a reservation): use the
>   **two-job** `slurm/launch_servers_polaris.sh` (multi-node, multi-model).

**Always run the probe first:** `qsub slurm/probe_vllm_polaris.pbs` proves the
toolchain (nvidia-smi → torch sees the A100 → vLLM loads the 0.5B and generates).
Don't run the smoke until the probe is green.

### Single-job smoke (debug)

```bash
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/decrypto
qsub slurm/smoke_polaris.pbs
tail -f logs/paper/polaris_smoke_<jid>.log     # watch for ">>> vLLM healthy" then "SMOKE GREEN"
```

It starts `vllm serve` in the background, waits for health, registers it via
`DECRYPTO_SERVERS_FILE` (with the node's HSN IP — real TCP), runs one self-play
episode, writes `results/polaris_smoke/experiment_summary.csv`, then stops vLLM.
Override the model with `qsub -v MODEL_KEY=...,MODEL_PATH=...,TP=... slurm/smoke_polaris.pbs`.

### Two-job launcher (prod / reservation, multi-node)

1. **Pick the model.** Edit the `models` / `ngpus` arrays near the top of
   `slurm/launch_servers_polaris.sh` (`"model_key:model_path"`; `ngpus` = TP size):

   ```bash
   models=( "qwen2.5_0.5B:$BASE/models/Qwen2.5-0.5B-Instruct" )
   ngpus=(1)
   ```

2. **Point the config at the same model.** In `config/examples/local_polaris.yaml`,
   keep `fixed_interceptor`, `models[].model_key`, and `models[].model_id`
   consistent. `model_id` must equal the name vLLM serves under — we serve under
   the **local path**, so `model_id` = the path.

3. **Submit** `bash slurm/launch_servers_polaris.sh` from the repo root. It prints
   each server job id + port, the per-launch `SERVERS_DIR`, and the dependent
   experiment job id, then the experiment job waits for health, runs the games,
   writes `results/polaris_smoke/experiment_summary.csv`, and `qdel`s the servers.
   (On debug it will refuse and tell you to use `smoke_polaris.pbs`.)

---

## What changed from the Midway (Slurm) scripts

Rather than edit the Midway scripts in place, we wrote PBS-flavored copies
(`*_polaris.*`) so both ports stay diffable. The substantive changes:

- **Scheduler.** `sbatch`→`qsub`; `#SBATCH`→`#PBS`; `--account=rcc-staff`→
  `-A lighthouse-uchicago`; `--partition=test`→`-q debug`;
  `--gres=gpu:N --nodes=1`→`-l select=1:ncpus=64:ngpus=4`; added the mandatory
  `-l filesystems=home:eagle`; `scancel`→`qdel`; `--dependency=after:`→
  `-W depend=after:`. PBS has **no `--wrap`**, so the inline Midway server
  command became a real script, `slurm/server_vllm_polaris.pbs`.
- **Server discovery (the trickiest port).** Midway stuffs `model_key:port` into
  the Slurm **job name** and recovers the node from `squeue %N`. PBS job names
  can't carry `:`, so we use the scheduler-agnostic **`DECRYPTO_SERVERS_FILE`**
  escape hatch in `src/utils/server.py` instead: each server job writes its own
  `{model_key, model_id, urls, job_ids}` JSON (with its **HSN-routable address**)
  into a per-launch directory once healthy; the experiment job points
  `DECRYPTO_SERVERS_FILE` at that directory. `server.py` was extended so that env
  var may be a **file or a directory of `<jobid>.json` files** (merged by
  `model_key`, half-written files skipped) — **the Slurm `squeue` path is
  untouched**. `ping_servers.py` / `runner.py` call `get_available_servers()`
  unchanged.
- **Environment.** Midway's `module load python/miniforge-25.3.0 && mamba
  activate …` → Polaris `module use /soft/modulefiles && module load
  conda/2025-09-25 && conda activate base`, then a **manually-activated**
  relocated venv staged from a tarball onto `/local/scratch` (see One-time setup).
  `vllm serve` is launched as `python -m vllm.entrypoints.openai.api_server`
  (the `vllm` console script's shebang hardcodes the eagle build path and would
  bypass the relocated python).
- **Paths.** All `/project/rcc/mehta5/...` → `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/...`.
- **Offline + caches.** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, `HF_HOME` on
  eagle, serve a **local model path** — compute nodes have no internet.
- **GPU memory.** A100 40 GB vs H200 140 GB: start small (Qwen2.5-0.5B, TP=1) and
  tune `--gpu-memory-utilization` when scaling to 70B (TP=4).

---

## Deliverables (this port)

| File | Role |
|---|---|
| `slurm/build_decrypto_serve_venv.sh` | one-time: build the serving venv + tarball on a login node |
| `slurm/probe_vllm_polaris.pbs` | single-GPU toolchain probe (run + pass first) |
| `slurm/smoke_polaris.pbs` | **single-job** smoke (server + experiment, one node) — the `debug`-queue path |
| `slurm/server_vllm_polaris.pbs` | per-model vLLM server job; registers itself via `DECRYPTO_SERVERS_FILE` (two-job path) |
| `slurm/launch_servers_polaris.sh` | two-job `qsub` launcher: server job(s) + dependent experiment job (prod/reservation) |
| `slurm/run_exp_polaris.pbs` | dependent experiment job: poll `ping_servers`, run `run.py`, `qdel` servers (two-job path) |
| `config/examples/local_polaris.yaml` | self-play, 1 episode, `match_encoder_decoder`, one `LocalModel` |
| `src/utils/server.py` | `DECRYPTO_SERVERS_FILE` may now be a dir of per-job JSON files (Slurm path intact) |

---

## Carry-over gotchas (from Midway / MARSHAL — they apply here)

- **`--enforce-eager` for the smoke** — skips `torch.compile`, avoiding the
  autotune-cache write that killed the engine on load at Midway.
- **Scrub inherited `TMPDIR`** → set it under `/local/scratch/$USER_<jobtag>`.
- **Do NOT `pip install -r requirements.txt`** — it pins torch 2.9 / vllm 0.13
  (unvalidated here). Use the proven cu124 set in `build_decrypto_serve_venv.sh`.
- **`transformers<5` + `tokenizers<0.22`** — a fresh install pulls transformers 5,
  which dropped `all_special_tokens_extended` that vLLM 0.8.x calls.
- **per-job cgroup `pids.max = 4096`** — bit MARSHAL's many-process Ray training
  hard, but `vllm serve` is light (one engine + one worker/GPU) and should fit. If
  you see `pthread_create … Resource temporarily unavailable`, cap the BLAS/OMP
  thread family (`OMP_NUM_THREADS=8`, etc.).
- **Polaris prologue/filesystem flakiness** — jobs can sit 20–40 min from `R` to
  script start, or hang the whole walltime with 0-byte output during an ALCF
  incident. If a job is silent, check `pbsnodes -l` for offlined nodes before
  assuming your bug. Every script streams a live `…wrap.log` so you can watch.
- **The ALCF proxy hangs in-cluster HTTP** (NEW, bit us at smoke jid 7186962).
  Polaris **compute** nodes export `http_proxy=http://proxy.alcf.anl.gov:3128`
  (via `/etc/sysconfig/proxy`) for outbound internet. `curl` AND the OpenAI/httpx
  client honor it, so a request to `localhost:<port>` or an HSN `10.201.x` address
  gets routed through the (air-gapped, unreachable) proxy and **hangs** — vLLM was
  up with `Application startup complete`, yet the health check timed out after
  900s. **Fix:** in every job `unset http_proxy https_proxy HTTP_PROXY
  HTTPS_PROXY all_proxy …` and set `no_proxy` to cover `localhost,127.0.0.1,
  10.201.0.0/16,.hsn.cm.polaris.alcf.anl.gov` (we never need outbound — fully
  offline). Health checks also use `curl --noproxy '*'` + a `urllib` ProxyHandler({})
  fallback, and hit `127.0.0.1` (not `localhost`, to dodge an IPv6 `::1` detour).

---

## Results so far

<!--RESULTS-->
- **Probe (jid 7186959, x3106c0s19b1n0):** GREEN. torch saw the A100, vLLM 0.8.4
  loaded Qwen2.5-0.5B (V1 engine) and generated. `Exit_status=0`.
- **Smoke (jid 7186966, x3106c0s13b0n0):** GREEN. The full orchestration path
  closed — `vllm serve` up → `DECRYPTO_SERVERS_FILE` discovery → `ping_servers`
  healthy over the HSN IP (0.55 s) → `run.py` ran 1 episode → `experiment_summary.csv`
  written → vLLM stopped. Episode: 2 turns, Eve intercepted, as expected for a
  0.5B model that fails JSON-format retries (model capacity, not pipeline).

The takeaway matches Midway: the orchestration path works on Polaris; real runs
just need a larger model (scale to Qwen2.5-72B / Llama-3.1-70B at TP=4, tuning
`--gpu-memory-utilization` for the 40 GB A100s — pre-stage the model first).
<!--/RESULTS-->

---

## Decisions / changes log

- **2026-06-07 — Orientation.** Confirmed Polaris login (`polaris-login-04`, PBS).
  Verified account **`lighthouse-uchicago`** (`sbank`: alloc 12374, ~17,175
  node-h), `debug` queue present. Confirmed the **self-hosted serving** task (NOT
  the Globus inference gateway). Read the Midway trio + `src/utils/server.py` and
  MARSHAL's `polaris_pbs_notes.md` (proven cluster facts + the venv-on-Lustre /
  air-gapped-compute / pids.max gotchas).
- **2026-06-07 — Env strategy.** The runner imports `anthropic` + `litellm`
  (`role_client.py`) and `gensim` (`embedding_baseline.py`) **unconditionally**,
  so even a LocalModel-only smoke needs them. The MARSHAL venv has the proven
  serving stack + most runner deps but is missing `litellm/anthropic/gensim/nltk`,
  and is MARSHAL's training env (left untouched). Built a dedicated
  **`decrypto-serve`** venv (`slurm/build_decrypto_serve_venv.sh`) reusing the
  proven cu124 pins + the pip cache, and packed it to a tarball for node-local-SSD
  staging.
- **2026-06-07 — Discovery design.** Used the `DECRYPTO_SERVERS_FILE` escape hatch
  rather than fighting `qstat` (PBS job names can't carry `model_key:port`).
  Extended `_load_servers_from_file` to accept a **directory** of per-job JSON
  files (merge by `model_key`, skip half-written files); server jobs publish their
  own `<jobid>.json` atomically (tmp + `os.replace`) once healthy, with the node's
  **HSN IP** (`10.201.x.x`, resolved from `<node>.hsn.cm.polaris.alcf.anl.gov`) so
  the experiment job on a different node can reach it. Slurm `squeue` path
  untouched; verified the new loader with a unit test (dir-merge, skip-partial,
  single-file forms).
- **2026-06-07 — Authored the Polaris scripts + config.** `probe_vllm_polaris.pbs`,
  `server_vllm_polaris.pbs`, `launch_servers_polaris.sh`, `run_exp_polaris.pbs`,
  `config/examples/local_polaris.yaml`. Confirmed `python -m
  vllm.entrypoints.openai.api_server` accepts all flags used
  (`--model/--host/--port/-tp/--gpu-memory-utilization/--max-model-len/
  --enforce-eager/--trust-remote-code/--disable-log-stats`). All scripts pass
  `bash -n`.
- **2026-06-07 — Probe GREEN (jid 7186959, node x3106c0s19b1n0).** ~80 min queue
  wait (debug contention), then ran in ~1 min: torch saw the A100 (driver
  570.124.06 / CUDA 12.8), vLLM 0.8.4 loaded Qwen2.5-0.5B on the **V1 engine**
  (Flash Attention, KV cache 782k tokens) and generated text; `Exit_status=0`.
  The tarball staged to `/local/scratch/.../decrypto-serve` and the relocated
  python ran cleanly. Note: vLLM logs a benign `Failed to get the IP address,
  using 0.0.0.0 by default` (the air-gapped-compute symptom MARSHAL hit) — vLLM
  falls back to 0.0.0.0, which is what we want. **V1 is fine for serving** (the
  MARSHAL V1 bug was about live weight-sync, not load-once serving).
- **2026-06-07 — Verified cross-node HSN addressing.** A compute node's
  `<node>.hsn.cm.polaris.alcf.anl.gov` resolves to `10.201.x.x` (dual HSN NICs);
  the server registers that IP and binds `0.0.0.0`, so the experiment job on a
  different node can reach it.
- **2026-06-07 — BLOCKER: the two-job dependent pattern can't run on `debug`.**
  `bash slurm/launch_servers_polaris.sh` submitted the server (jid 7186960) but
  PBS rejected the dependent experiment job: `would exceed queue generic's
  per-user limit of jobs in 'Q' state`. `qstat -Qf debug`:
  **`max_run=[u:PBS_GENERIC=1]`** and **`queued_jobs_threshold=[u:PBS_GENERIC=1]`**
  — a user may have only 1 running AND 1 queued job on debug. So a server job and
  a separate experiment job can never be Running together (and since the
  experiment `qdel`s the server, two jobs would deadlock). `qdel`'d the orphan
  server. **Fix:** added `slurm/smoke_polaris.pbs`, a single-job (one node)
  server+experiment smoke for debug; kept the two-job launcher for queues that
  allow ≥2 concurrent jobs and made it `qdel` its server + point at
  `smoke_polaris.pbs` if the dependent submit is rejected. (debug also caps
  `nodect=2`, so a single 2-node job could host server+exp on separate nodes — a
  future option if literal cross-node routing must be exercised on debug.)
- **2026-06-07 — Submitted single-job smoke** `qsub slurm/smoke_polaris.pbs`
  (jid 7186962, debug).
- **2026-06-07 — Smoke 7186962 FAILED: ALCF proxy hangs the health check (not a
  vLLM problem).** Started immediately on x3106c0s19b1n0; venv staged in 9s, but
  `vLLM not healthy in 900s` (Exit_status 45). The vLLM server log showed it came
  up cleanly in ~30s (`Application startup complete`, API on `http://0.0.0.0:8305`,
  `/health` route live) — so the **health check** was wrong, not vLLM. Root cause:
  compute nodes set `http_proxy=proxy.alcf.anl.gov:3128`, so `curl localhost:8305`
  routed through the unreachable proxy. **Fix:** unset all proxy vars + set
  `no_proxy` in every job; proxy-immune health check (`curl --noproxy '*'` + urllib
  ProxyHandler({}), hitting 127.0.0.1) with a one-time proxy/socket diagnostic.
  Resubmitted as jid **7186966**.
- **2026-06-07 — SMOKE GREEN (jid 7186966, node x3106c0s13b0n0).** With the proxy
  bypass: venv staged in 12 s, vLLM healthy in **21 s**, server registered with its
  HSN IP `http://10.201.3.1:8159/v1`, `ping_servers` got a healthy reply in 0.55 s
  (file-based discovery + real HTTP over the HSN NIC), `run.py +get_models_from_slurm=true`
  discovered the server and completed **1/1** episodes, wrote
  `results/polaris_smoke/experiment_summary.csv`, and the cleanup trap stopped vLLM.
  `Exit_status=0`. The 0.5B model's gameplay is garbage (JSON-retry failures) — a
  model-capacity issue, not a pipeline one, matching Midway. **The self-hosted
  vLLM serving pipeline is proven on Polaris.**
