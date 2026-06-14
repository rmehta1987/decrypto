# Setting up the Decrypto pipeline on ALCF Polaris — step by step

A procedural, copy-paste guide for standing up the **Decrypto self-hosted vLLM
serving pipeline** on ALCF **Polaris** (PBS Pro): jobs run `vllm serve` on
A100 nodes, and the Decrypto game loop drives them over HTTP. Follow the
numbered steps top to bottom for a first green run; the two sections at the end
cover the multi-model cross-play experiment and long production sessions.

- Sub-bullets marked **Did not work / gotcha** record the dead ends we hit so
  you don't repeat them. The blow-by-blow rationale and the complete job ledger
  live in [`polaris_pbs_notes.md`](polaris_pbs_notes.md) — this file is the
  *recipe*, that one is the *lab notebook*.

> **Scope:** self-hosted serving only. This is **not** the ALCF
> inference-*gateway* path (Globus-authed HTTPS calls to models ALCF already
> hosts); that work lives on the `alcf-inference-gateway` branch and none of its
> files exist on this branch.

---

## Cluster facts (quick reference)

| Item | Value |
|---|---|
| Scheduler | **PBS Pro** — `qsub` / `qstat` / `qdel` |
| Account (`-A`) | **`lighthouse-uchicago`** (NOT "Uchicago-lighthouse" — PBS rejects it) |
| Queue (`-q`) | **`debug`** for probes/smokes; **`preemptable`** for multi-job runs. See the [queue table](#polaris-queues-verified-2026-06-12) |
| GPU | **4× NVIDIA A100-SXM4-40GB** per node (sm_80). Verified on-node via `nvidia-smi` (job 7197265) and the ALCF compute-nodes doc; there is no 80 GB partition. Much tighter than Midway's H200 (~140 GB) |
| CPU / RAM | AMD EPYC Milan, 64 threads (`ncpus=64`), 512 GiB |
| Required select line | `-l select=1:ncpus=64:ngpus=4` (do **not** add `:system=polaris`) |
| Required filesystems | `-l filesystems=home:eagle` (PBS rejects jobs that omit it) |
| Conda module | `module use /soft/modulefiles && module load conda/2025-09-25 && conda activate base` |
| Member base (`$BASE`) | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5` |
| Project root | `$BASE/decrypto` |
| Model store | `$BASE/models/` |
| Serving venv + tarball | `$BASE/conda-envs/decrypto-serve` + `$BASE/decrypto-serve-venv.tar` |
| Node-local scratch | `/local/scratch` (RAID0 SSD; falls back to `/tmp`) — venv staging + `TMPDIR` |
| Bare job id | `${PBS_JOBID%%.*}` (full id is `1234567.polaris-pbs-01.hsn…`) |

Throughout, `$BASE` = `/lus/eagle/projects/lighthouse-uchicago/members/mehta5`.

### Polaris queues (verified 2026-06-12)

Limits confirmed with `qstat -Qf <queue>` on the cluster and cross-checked
against the [ALCF running-jobs docs](https://docs.alcf.anl.gov/polaris/running-jobs/).
"Nodes" is the per-job min–max; the concurrency limit is what gates the
**two-job** pattern (it needs **≥2 jobs running at once**). Valid `filesystems`
values are `home` and `eagle`.

| Queue | Nodes/job | Walltime | Concurrency limit | Fits our use |
|---|---|---|---|---|
| `debug` | 1–2 | 5 min – 1 h | **1 running + 1 queued / user** | probe + single-job smoke |
| `debug-scaling` | 1–10 | 5 min – 1 h | 1 running/queued / user | single-job only |
| `prod` (routing) | **10–496** | 5 min – 24 h | 10 running / project | two-job, but ≥10 nodes/job — rejects our 1-node jobs |
| `preemptable` | 1–10 | 5 min – **72 h** | 10 running / project, 20 queued / user | **the multi-job queue** (3 servers + 1 experiment). Preemptible — submit with `-r y` |
| `capacity` | 1–4 | 5 min – **168 h** | **1 running**, 2 total / user | long single-job runs only |
| `demand` | 1–56 | 5 min – 1 h | by request only | not applicable |

- **Two-job / multi-job pattern** (servers + dependent experiment, ≥2 running)
  → `preemptable`. Jobs there can be killed without warning when `demand` work
  arrives: submit with `-r y` (the launcher does) and write results
  incrementally.
- **Single-job pattern** (server + experiment fused) → `debug` for the smoke;
  `capacity` could host a long no-preemption run but only by fusing all servers
  + experiment into one multi-node job (not implemented).

---

## Step 0 — Log in and confirm the allocation

```bash
ssh polaris.alcf.anl.gov            # lands on a login node (has internet)
sbank-list-allocations              # confirm: alloc 12374 lighthouse-uchicago, node-h > 0
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/decrypto
```

- **Did not work — the account name.** `-A Uchicago-lighthouse` is rejected at
  submit time. The only accepted form is **`-A lighthouse-uchicago`** (already
  baked into every `#PBS` header).

---

## Step 1 — Build the serving venv + tarball (one-time, login node)

One env hosts **both** the vLLM server and the HTTP-only game runner. Build it on
a login node (needs internet), then pack it into a single tarball for fast
staging.

```bash
bash slurm/build_decrypto_serve_venv.sh
```

This produces `$BASE/conda-envs/decrypto-serve` and packs it to
`$BASE/decrypto-serve-venv.tar`. The proven cu124 stack:

```
torch 2.6.0+cu124 · vllm 0.8.4 · transformers 4.51.2 · tokenizers <0.22 · numpy <2
+ runner: litellm anthropic gensim nltk hydra-core openai python-dotenv scipy pandas tqdm requests omegaconf
```

- **Did not work — `pip install -r requirements.txt`.** It pins torch 2.9 /
  vllm 0.13, which are unvalidated against the Polaris driver/CUDA. Use the
  script's pinned set.
- **Did not work — an unpinned install.** A fresh install pulls `transformers` 5,
  which dropped `all_special_tokens_extended` that vLLM 0.8.x calls → load-time
  `AttributeError`. The build script re-pins `transformers==4.51.2` /
  `tokenizers<0.22` **after** the runner deps in case one of them dragged it
  forward.
- **Did not work — running a plain venv straight off eagle.** Lustre is fast for
  big sequential reads but chokes on the ~70k-tiny-file metadata storm that
  `import torch`/vllm triggers on a cold compute node (MARSHAL measured ~19 min /
  effective hang). The tarball sidesteps this: one big file, extracted locally.
- **Gotcha — the runner is not optional-import.** `role_client.py` imports
  `anthropic` + `litellm` and `embedding_baseline.py` imports `gensim`
  **unconditionally**, so even a LocalModel-only smoke needs them in the env.

---

## Step 2 — Pre-download every model you'll serve (one-time, login node)

**Compute nodes are offline** — stage weights to eagle from a login node, then
serve the **local path**. Staged today (verification: `config.json` present and
every shard in `model.safetensors.index.json` exists on disk):

| Path under `$BASE/models/` | HF repo | Size |
|---|---|---|
| `Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | 954 MB |
| `Qwen3-8B` | `Qwen/Qwen3-8B` | 16 GB |
| `Qwen3-4B` | `Qwen/Qwen3-4B` | 7.5 GB |
| `Qwen2.5-72B-Instruct` | `Qwen/Qwen2.5-72B-Instruct` | ~145 GB (37 shards) |

(The 70B-class slot was originally `meta-llama/Meta-Llama-3.1-70B-Instruct`;
that repo is gated, no HF token existed here — 401 `GatedRepoError`,
2026-06-12 — and the owner substituted the open Qwen2.5-72B-Instruct.)

```bash
# in the decrypto-serve venv, on a login node:
huggingface-cli download Qwen/Qwen3-8B --local-dir $BASE/models/Qwen3-8B
```

- **Gotcha — gated repos need a token.** `meta-llama/*` requires an accepted
  Meta license on your HF account plus a token on this machine
  (`huggingface-cli login` or `HF_TOKEN`). Without one the download fails
  immediately with `401 GatedRepoError` — that is the cause, not a bug
  (observed 2026-06-12).
- **Gotcha — check disk before a 100+ GB pull** (`df -h /lus/eagle/...` and
  `du -sh $BASE/models`); do not discover mid-download that eagle is full.
- **Did not work — serving an HF repo id.** Jobs set `HF_HUB_OFFLINE=1` /
  `TRANSFORMERS_OFFLINE=1`; a repo id would try to hit the Hub and fail. The
  config's `model_id` must be the **local path**, because that's the name vLLM
  serves the model under.
- **Gotcha — verify the download.** A truncated pull is a silent failure: list
  the dir and confirm `config.json`, `model.safetensors.index.json`, and every
  shard named in the index are present.

---

## Step 3 — Run the toolchain probe (prove the stack before anything bigger)

Always probe first. It checks `nvidia-smi → torch sees the A100 → vLLM loads
each model at its TP and generates`, in isolation.

```bash
qsub slurm/probe_vllm_polaris.pbs          # default: the 0.5B at TP=1
tail -f logs/probe_wrap_<jid>.log          # watch live

# probe several models in ONE debug job (entries are path:tp:mem_util:max_len,
# separated by ';' — qsub -v eats commas):
qsub -l walltime=00:40:00 \
  -v MODEL_SPECS="$BASE/models/Qwen3-8B:1:0.90:8192;$BASE/models/Qwen3-4B:1:0.90:8192" \
  slurm/probe_vllm_polaris.pbs
```

The probe prints each model's KV-cache line (`GPU KV cache size: N tokens`) —
capture it; it is the evidence your TP choice left usable KV space.

- **Did not work — trusting the `.OU`/`.ER` files mid-run.** PBS only flushes
  them at job *end*, which hides early hangs. Every script mirrors output to a
  live `…wrap.log` on eagle — `tail -f` that instead.
- **Note — benign vLLM log line.** `Failed to get the IP address, using 0.0.0.0
  by default` is expected on the air-gapped node; vLLM falls back to `0.0.0.0`,
  which is what we want.
- **Note — V1 engine is fine.** vLLM loads on the V1 engine here; that's correct
  for load-once serving (the MARSHAL V1 issue was about live weight-sync, not
  serving).
- **Gotcha — queue wait varies wildly.** The first probe sat ~80 min in `Q`
  (debug contention); the 2026-06-12 probe started in seconds. Don't assume a
  queued job is stuck; check `qstat -u $USER`.

---

## Step 4 — Run the smoke test (single job, `debug` queue)

This is the **single-job** path: one job starts `vllm serve` in the background,
waits for health, registers it for discovery, runs **one** self-play episode,
writes results, and stops vLLM.

```bash
qsub slurm/smoke_polaris.pbs
tail -f logs/paper/polaris_smoke_<jid>.log   # ">>> vLLM healthy" → … → "SMOKE GREEN"
```

Result lands at `results/polaris_smoke/experiment_summary.csv`. Override the
model inline: `qsub -v MODEL_KEY=...,MODEL_PATH=...,TP=... slurm/smoke_polaris.pbs`.

- **Did not work — the faithful two-job pattern on `debug`.** `debug` enforces
  **`max_run=1`** *and* **`queued_jobs_threshold=1`** per user (`qstat -Qf debug`).
  A separate server job + dependent experiment job can therefore never both be
  Running — and since the experiment `qdel`s the server, they'd deadlock. That's
  the whole reason `smoke_polaris.pbs` fuses server + experiment into one job.
- **Did not work — in-cluster HTTP with the ALCF proxy set (this bit us hard).**
  Compute nodes export `http_proxy=proxy.alcf.anl.gov:3128`. `curl` *and* the
  OpenAI/httpx client honor it, so a request to `localhost:<port>` or an HSN
  `10.201.x` address gets routed through the (air-gapped, unreachable) proxy and
  **hangs** — vLLM was up with `Application startup complete`, yet the health
  check timed out after 900 s (smoke 7186962). Fix (already in every script):
  `unset http_proxy https_proxy …`, set `no_proxy` to cover
  `localhost,127.0.0.1,10.201.0.0/16,.hsn.cm.polaris.alcf.anl.gov`, do health
  checks with `curl --noproxy '*'` + a `urllib` `ProxyHandler({})` fallback, and
  hit `127.0.0.1` (not `localhost`, to dodge an IPv6 `::1` detour).
- **Did not work — inheriting `TMPDIR`.** Every script `unset TMPDIR` and sets it
  under `/local/scratch/${USER}_<jobtag>`, so a stale path from the submitting
  shell can't break the torch.compile/inductor cache writes.
- **Did not work — launching via the `vllm` console script.** Its shebang
  hardcodes the eagle build path and would bypass the relocated venv. Launch with
  `python -m vllm.entrypoints.openai.api_server` instead.
- **Note — 0.5B gameplay being garbage is expected.** Qwen2.5-0.5B fails the
  JSON-format retries, so the *episode* is nonsense. That's a model-capacity
  issue, **not** a pipeline issue — the smoke proves orchestration, not play
  quality.

---

## Step 5 — The multi-model cross-play run (`preemptable`, multi-node)

The realized experiment: **Qwen2.5-72B-Instruct + Qwen3-8B + Qwen3-4B** served
concurrently, all three models available to all three Decrypto roles → the
runner's `itertools.product` yields **27 encoder×decoder×interceptor
combinations per env seed** (`config/examples/local_polaris_3model.yaml`:
`match_encoder_decoder: false`, no `fixed_*`, empty `filter_model`).

### Tensor parallelism (decided by memory + head divisibility, verified on-cluster)

Per-GPU weights ≈ `2 × params / TP` bytes (bf16); TP must divide both
`num_attention_heads` and `num_key_value_heads` (read from each staged
`config.json` — do not trust memory).

| Model | bf16 weights | Heads (attn/KV) | TP | weights/GPU | KV-cache evidence |
|---|---|---|---|---|---|
| Qwen2.5-72B | ~145 GB | 64 / 8 (verify from staged config.json) | **4** | ~36.4 GB — very tight on 40 GB; launcher uses mem_util 0.95 | pending the 72B probe; if it OOMs or KV < max_len, drop `--max-model-len` 8192→4096→2048 |
| Qwen3-8B | ~16 GB | 32 / 8 | 1 | ~16 GB | job 7197265: 133,312 tokens @ max_len 8192 |
| Qwen3-4B | ~8 GB | 32 / 8 | 1 | ~8 GB | job 7197265: 189,648 tokens |

- **Single-node TP only.** Multi-node TP/PP needs Ray spanning nodes — a real
  bring-up, out of scope. 70B must fit one node at TP=4.
- One model per node-job: 3 server jobs + 1 experiment job = **4 jobs / 4 nodes**
  concurrent (the experiment node is CPU/HTTP-only; PBS hands out whole nodes).
  Co-locating 8B+4B on one node (TP=1 each, `CUDA_VISIBLE_DEVICES=1` + distinct
  port for the second server) would drop it to 3 — deferred until the simple
  layout is proven.

### Launching — use the FUSED single job (the proven path)

```bash
# 27-game orchestration smoke (1 seed × 1 episode), one 4-node job:
qsub slurm/fused_3model_polaris.pbs

# production run (15 seeds = 405 games), capacity queue (no preemption, ≤168 h):
qsub -q capacity -l walltime=16:00:00 \
  -v EXP_NAME=polaris_3model,SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14" \
  slurm/fused_3model_polaris.pbs
```

`fused_3model_polaris.pbs` takes **one 4-node job**: node 0 runs the experiment
(CPU/HTTP only), nodes 1–3 each run one `vllm serve` via `mpiexec --hosts`
(per-node bootstrap `slurm/start_vllm_server_node.sh`, same staging/proxy/
discovery logic as the proven server job). One queue wait, no inter-job skew,
atomic lifetime; job 7197574 validated it end-to-end (3 servers ready in 361 s,
all 27 combos, 25 min total).

- **Did not work — the two-job dependent pattern on a contended queue.** The
  launcher (`launch_servers_polaris.sh` + `run_exp_polaris.pbs`) is kept and
  works mechanically, but on the busy `preemptable` queue the server jobs and
  the experiment job are scheduled **independently**: in the 2026-06-12
  attempts the servers started 3–7 h before the experiment could get a node and
  **died at exactly their walltime while idle** (jobs 7197359/7197380/7197381,
  `Exit_status=-29`). `-W depend=after:` orders starts; it does not co-schedule.
  If you must use the two-job form, set server walltime ≫ experiment walltime +
  worst-case queue skew (hours), and accept the idle-node waste.
- **Did not work — recovering a mid-run preempted server.** `-r y` does requeue
  a preempted server (observed: run_count=2), but the rerun registers a **new**
  HSN address, and the runner reads server URLs **once at startup** — so a
  preemption mid-experiment permanently breaks that model's games. This is why
  the production run belongs on `capacity` (no preemption), not `preemptable`.

- **How discovery works (no `qstat` parsing).** PBS job names can't carry
  `model_key:port`, so each server job writes its own
  `{model_key, model_id, urls, job_ids}` JSON — with its **HSN-routable IP**
  (`10.201.x.x`) — into a per-launch `SERVERS_DIR`. The experiment job points
  `DECRYPTO_SERVERS_FILE` at that directory; `get_available_servers()` merges the
  per-job files. (The Slurm `squeue` discovery path in `server.py` is untouched.)
- **Keep `model_key` / `model_id` consistent in both places** or discovery
  silently finds nothing: the launcher's `models` array and the config's
  `models[].model_key` + `models[].model_id` (the latter must equal the served
  local path).
- **Gotcha — `qsub -v` splits on commas.** List-valued parameters must avoid
  `,`: the probe's `MODEL_SPECS` uses `;` between entries, and the experiment's
  `SEEDS` is space-separated (`SEEDS="0 1 2 3 4"`), rebuilt into a Hydra list
  inside the job.
- **Gotcha — Qwen3 models think.** Qwen3-8B/-4B emit `<think>` blocks; with a
  small `max_tokens` they burn the whole budget thinking and never emit the JSON
  the runner parses (looks like a pipeline bug; it is a token-budget issue). The
  3-model config bounds it with `max_tokens: 2500` + `max_reasoning_tokens: 2000`
  (two-stage pattern in `role_client.py`).
- **Did not work — running this launcher on `debug`.** The dependent `qsub` is
  rejected (`max_run=1`/`queued=1`); the launcher detects that, `qdel`s the
  orphan servers so they don't burn allocation alone, and tells you to use
  `smoke_polaris.pbs`.
- **Gotcha — raise the timeouts for 70B.** A 70B cold load at TP=4 reads ~141 GB
  off Lustre — minutes, not the 0.5B's 21 s. The launcher passes
  `HEALTH_TIMEOUT=2400` (server-side health wait) and `WAIT_TIMEOUT=3600`
  (experiment-side ready poll) by default.

---

## Running a longer session

The orchestration smoke is 27 games (1 seed × 1 episode). A production run
scales seeds/episodes and the wall clock.

- **Scale via env knobs**, not config edits: `SEEDS="0 1 2 ..."` widens
  `env_seed` (27 combos × |seeds| games at `num_episodes: 1`), `NUM_EPISODES=N`
  repeats each combo×seed. Mind the blow-up and note the count you ran.
- **Cap concurrency to the slowest server — `DECRYPTO_MAX_WORKERS`.** The runner
  spawns **one process per game** (`ProcessPoolExecutor(max_workers=total
  games)`). Run all 405 games at once and they flood the slow 72B server
  (KV cache ~18,800 tokens ≈ 2.3× concurrency): job 7199012 lost ~40% of games —
  almost all of them 72B-involving — when overloaded requests hit
  `APITimeoutError`, returned `None`, and crashed the answer parser; it wrote
  only 245/405 rows (610 aborted requests on the 72B server). The fused script
  sets `DECRYPTO_MAX_WORKERS=24` (below the 27-game smoke's proven-safe load) so
  the 72B is never overwhelmed; the run trades wall-clock (~2.5 h vs ~1.5 h) for
  completeness. **Match game concurrency
  to the slowest server's KV capacity, not the combination count.** Default
  (env unset) preserves the original one-worker-per-game behavior.
- **`pids.max=4096`.** With `OMP_NUM_THREADS=1` (set by the fused script) and
  `DECRYPTO_MAX_WORKERS=24`, process/thread counts stay far under the per-job
  cgroup cap regardless of seed count.
- **Queue: `capacity` for production.** No preemption, ≤168 h, 1–4 nodes — the
  fused 4-node job fits exactly, and it started within a minute on 2026-06-12
  while preemptable 1-node jobs waited 3–7 h. Walltime is an upper bound; the
  job exits when the matrix completes.
- **Recoverability:** per-game result dirs land incrementally under
  `results/<EXP_NAME>/` as games finish (the summary CSV is written at the
  end), so a walltime kill loses only in-flight games; a rerun restarts the
  matrix.
- **Gotcha — `prod` won't take a single-node job.** It's a routing queue with a
  **10-node-per-job minimum**; only reach for it if you genuinely run ≥10-node
  jobs.
- **Consider dropping `--enforce-eager` once a model is stable** for a throughput
  win (it currently skips `torch.compile`). The cache dirs already live on eagle
  (`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`), so the autotune write that
  killed the engine at Midway is mitigated — but re-verify on a probe before a
  long run.
- **Watch thread/process limits.** The per-job cgroup caps `pids.max = 4096`.
  `vllm serve` is light (one engine + one worker/GPU) so it fits, but if you see
  `pthread_create … Resource temporarily unavailable`, cap the BLAS/OMP thread
  family (`OMP_NUM_THREADS=8`, etc.).
- **Gotcha — prologue / filesystem flakiness on long jobs.** Jobs can sit
  20–40 min from `R` to script start, or hang the whole walltime with 0-byte
  output during an ALCF incident. If a job is silent, check `pbsnodes -l` for
  offlined nodes before assuming it's your bug — and `tail -f` the live
  `…wrap.log` to see real progress.

---

## Quick command reference

```bash
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/decrypto

# one-time (login node):
bash slurm/build_decrypto_serve_venv.sh            # build venv + tarball
huggingface-cli download <repo> --local-dir $BASE/models/<name>   # stage a model

# prove + smoke (debug queue, single node):
qsub slurm/probe_vllm_polaris.pbs                  # toolchain probe — run first
qsub -v MODEL_SPECS="<path>:<tp>:<mem>:<len>;..." slurm/probe_vllm_polaris.pbs
qsub slurm/smoke_polaris.pbs                       # server + 1 episode + results
qsub -v MODEL_KEY=...,MODEL_PATH=...,TP=4 slurm/smoke_polaris.pbs   # override model

# multi-model cross-play — fused single 4-node job (the proven path):
qsub slurm/fused_3model_polaris.pbs                # 27-game smoke (preemptable)
qsub -q capacity -l walltime=16:00:00 \
  -v EXP_NAME=polaris_3model,SEEDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14" \
  slurm/fused_3model_polaris.pbs                   # production (405 games)

# (two-job launcher exists but loses to queue skew on busy queues — see Step 5)

# watch / manage:
tail -f logs/paper/polaris_fused_<jid>.log         # fused job progress
tail -f logs/vllm/<model_key>-<jid>.wrap.log       # per-server progress
qstat -u $USER          # my jobs        qdel <jid>     # cancel
qstat -Qf capacity      # queue limits   pbsnodes -l    # offlined nodes
```
