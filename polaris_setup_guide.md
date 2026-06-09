# Setting up the Decrypto pipeline on ALCF Polaris — step by step

A procedural, copy-paste guide for standing up the **Decrypto self-hosted vLLM
serving pipeline** on ALCF **Polaris** (PBS Pro): one job runs `vllm serve` on an
A100 node, and the Decrypto game loop drives it over HTTP. Follow the numbered
steps top to bottom for a first green run.

- Sub-bullets marked **✗ Didn't work / gotcha** record the dead ends we hit so you
  don't repeat them. The blow-by-blow rationale lives in
  [`polaris_pbs_notes.md`](polaris_pbs_notes.md) (status banner + decisions log) —
  this file is the *recipe*, that one is the *lab notebook*.
- Two sections at the end — **Scaling to a bigger / more capable LLM** and
  **Running a longer session** — describe what changes once the smoke is green.

> **Scope:** self-hosted serving only. This is **NOT** the ALCF inference-*gateway*
> path (`argonne_polaris.yaml` / `inference_auth_token.py`), which calls models
> ALCF already hosts over HTTPS and is blocked on Globus.

---

## Cluster facts (quick reference)

| Item | Value |
|---|---|
| Scheduler | **PBS Pro** — `qsub` / `qstat` / `qdel`; login host `polaris-login-04` |
| Account (`-A`) | **`lighthouse-uchicago`** (⚠️ NOT "Uchicago-lighthouse" — PBS rejects it) |
| Queue (`-q`) | **`debug`** for the smoke (1–2 nodes, ≤1 h). See the [queue table](#polaris-queues-alcf-docs) for the rest |
| GPU | **4× NVIDIA A100 40 GB** per node (sm_80). Much tighter than Midway's H200 (~140 GB) |
| CPU / RAM | AMD EPYC Milan, 64 threads (`ncpus=64`), 512 GiB |
| Required select line | `-l select=1:ncpus=64:ngpus=4` (do **NOT** add `:system=polaris`) |
| Required filesystems | `-l filesystems=home:eagle` (PBS rejects jobs that omit it) |
| Conda module | `module use /soft/modulefiles && module load conda/2025-09-25 && conda activate base` |
| Member base (`$BASE`) | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5` |
| Project root | `$BASE/decrypto` |
| Model store | `$BASE/models/` |
| Serving venv + tarball | `$BASE/conda-envs/decrypto-serve` + `$BASE/decrypto-serve-venv.tar` |
| Node-local scratch | `/local/scratch` (RAID0 SSD; falls back to `/tmp`) — venv staging + `TMPDIR` |
| Bare job id | `${PBS_JOBID%%.*}` (full id is `1234567.polaris-pbs-01.hsn…`) |

Throughout, `$BASE` = `/lus/eagle/projects/lighthouse-uchicago/members/mehta5`.

### Polaris queues (ALCF docs)

From the [ALCF Polaris running-jobs docs](https://docs.alcf.anl.gov/polaris/running-jobs/).
"Nodes" is the per-job min–max; the concurrency limit is what gates the **two-job**
pattern (it needs **≥2 jobs running at once**). Valid `filesystems` values are
`home` and `eagle`.

| Queue | Nodes/job | Walltime | Concurrency limit | Fits our use |
|---|---|---|---|---|
| `debug` | 1–2 | 5 min – 1 h | **1 running + 1 queued / user** | probe + single-job smoke |
| `debug-scaling` | 1–10 | 5 min – 1 h | 1 running/queued / user | single-job only |
| `prod` (routing) | **10–496** | 5 min – 24 h | 10 running, 100 queued / project | two-job, but ≥10 nodes/job |
| `preemptable` | 1–10 | 5 min – **72 h** | 20 running/queued / project | **long two-job runs** (⚠ preemptible) |
| `capacity` | 1–4 | 5 min – **168 h** | **1 running**, 2 total / user | **long single-job runs** |
| `demand` | 1–56 | 5 min – 1 h | 100 / project | by request only |

- **Two-job pattern** (server + dependent experiment, ≥2 running) → `preemptable`
  (flexible node count; add `#PBS -r y` so a preempted job reruns), `demand`
  (by request), or `prod` (but every job must request ≥10 nodes).
- **Single-job pattern** (server + experiment fused in one job) → `debug` for the
  smoke; **`capacity`** for a long run (up to 7 days, no preemption, 1 running job).

---

## Step 0 — Log in and confirm the allocation

```bash
ssh polaris.alcf.anl.gov            # lands on a login node (has internet)
sbank-list-allocations              # confirm: alloc 12374 lighthouse-uchicago, node-h > 0
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/decrypto
```

- **✗ Gotcha — the account name.** `-A Uchicago-lighthouse` is rejected at submit
  time. The only accepted form is **`-A lighthouse-uchicago`** (already baked into
  every `#PBS` header).

---

## Step 1 — Build the serving venv + tarball (one-time, login node)

One env hosts **both** the vLLM server and the HTTP-only game runner. Build it on a
login node (needs internet), then pack it into a single tarball for fast staging.

```bash
bash slurm/build_decrypto_serve_venv.sh
```

This produces `$BASE/conda-envs/decrypto-serve` and packs it to
`$BASE/decrypto-serve-venv.tar`. The proven cu124 stack:

```
torch 2.6.0+cu124 · vllm 0.8.4 · transformers 4.51.2 · tokenizers <0.22 · numpy <2
+ runner: litellm anthropic gensim nltk hydra-core openai python-dotenv scipy pandas tqdm requests omegaconf
```

- **✗ Don't `pip install -r requirements.txt`.** It pins torch 2.9 / vllm 0.13,
  which are unvalidated against the Polaris driver/CUDA. Use the script's pinned set.
- **✗ A fresh install pulls `transformers` 5**, which dropped
  `all_special_tokens_extended` that vLLM 0.8.x calls → load-time `AttributeError`.
  The build script re-pins `transformers==4.51.2` / `tokenizers<0.22` **after** the
  runner deps in case one of them dragged it forward.
- **✗ Don't run a plain venv straight off eagle.** Lustre is fast for big
  sequential reads but chokes on the ~70k-tiny-file metadata storm that
  `import torch`/vllm triggers on a cold compute node (MARSHAL measured ~19 min /
  effective hang). The tarball sidesteps this: one big file, extracted locally.
- **✗ The runner is not optional-import.** `role_client.py` imports `anthropic` +
  `litellm` and `embedding_baseline.py` imports `gensim` **unconditionally**, so
  even a LocalModel-only smoke needs them in the env.

---

## Step 2 — Pre-download every model you'll serve (one-time, login node)

**Compute nodes are offline** — you must stage weights to eagle from a login node.
The 0.5B smoke model is already at `$BASE/models/Qwen2.5-0.5B-Instruct`.

```bash
# in the decrypto-serve venv, on a login node:
huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
  --local-dir $BASE/models/Qwen2.5-72B-Instruct
```

- **✗ Gotcha — serve a local *path*, never a HF repo id.** Jobs set
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`; a repo id would try to hit the
  Hub and fail. The config's `model_id` must be the **local path**, because that's
  the name vLLM serves the model under.

---

## Step 3 — Run the toolchain probe (prove the stack before the smoke)

Always probe first. It checks `nvidia-smi → torch sees the A100 → vLLM loads the
0.5B and generates`, in isolation.

```bash
qsub slurm/probe_vllm_polaris.pbs
tail -f logs/probe_wrap_<jid>.log          # watch live; ends with "Probe complete: GREEN"
```

- **✗ Don't trust the `.OU`/`.ER` files mid-run.** PBS only flushes them at job
  *end*, which hides early hangs. Every script mirrors output to a live
  `…wrap.log` on eagle — `tail -f` that instead.
- **Note — benign vLLM log line.** `Failed to get the IP address, using 0.0.0.0 by
  default` is expected on the air-gapped node; vLLM falls back to `0.0.0.0`, which
  is what we want.
- **Note — V1 engine is fine.** vLLM loads on the V1 engine here; that's correct for
  load-once serving (the MARSHAL V1 issue was about live weight-sync, not serving).
- **✗ Expect a queue wait.** `debug` is contended — the first probe sat ~80 min in
  `Q` before running ~1 min. Don't assume it's stuck; check `qstat -u $USER`.

---

## Step 4 — Run the smoke test (single job, `debug` queue)

This is the **single-job** path: one job starts `vllm serve` in the background,
waits for health, registers it for discovery, runs **one** self-play episode,
writes results, and stops vLLM.

```bash
qsub slurm/smoke_polaris.pbs
tail -f logs/paper/polaris_smoke_<jid>.log   # ">>> vLLM healthy" → … → "SMOKE GREEN"
```

Result lands at `results/polaris_smoke/experiment_summary.csv`. Override the model
inline: `qsub -v MODEL_KEY=...,MODEL_PATH=...,TP=... slurm/smoke_polaris.pbs`.

- **✗ The faithful two-job pattern can't run on `debug`.** `debug` enforces
  **`max_run=1`** *and* **`queued_jobs_threshold=1`** per user (`qstat -Qf debug`).
  A separate server job + dependent experiment job can therefore never both be
  Running — and since the experiment `qdel`s the server, they'd deadlock. That's
  the whole reason `smoke_polaris.pbs` fuses server + experiment into one job.
- **✗ The ALCF proxy hangs in-cluster HTTP (this bit us hard).** Compute nodes
  export `http_proxy=proxy.alcf.anl.gov:3128`. `curl` *and* the OpenAI/httpx client
  honor it, so a request to `localhost:<port>` or an HSN `10.201.x` address gets
  routed through the (air-gapped, unreachable) proxy and **hangs** — vLLM was up
  with `Application startup complete`, yet the health check timed out after 900 s.
  Fix (already in every script): `unset http_proxy https_proxy …`, set `no_proxy`
  to cover `localhost,127.0.0.1,10.201.0.0/16,.hsn.cm.polaris.alcf.anl.gov`, do
  health checks with `curl --noproxy '*'` + a `urllib` `ProxyHandler({})` fallback,
  and hit `127.0.0.1` (not `localhost`, to dodge an IPv6 `::1` detour).
- **✗ Inherited `TMPDIR` poisons the run.** Every script `unset TMPDIR` and sets it
  under `/local/scratch/${USER}_<jobtag>`, so a stale path from the submitting
  shell can't break the torch.compile/inductor cache writes.
- **✗ Don't launch via the `vllm` console script.** Its shebang hardcodes the
  eagle build path and would bypass the relocated venv. Launch with
  `python -m vllm.entrypoints.openai.api_server` instead.
- **Note — gameplay being garbage is expected.** Qwen2.5-0.5B fails the JSON-format
  retries, so the *episode* is nonsense. That's a model-capacity issue, **not** a
  pipeline issue — the smoke proves orchestration, not play quality.

---

## Step 5 — (Optional) the two-job launcher on a multi-job queue

For multi-node / multi-model runs on a queue that allows **≥2 concurrent jobs**
(`preemptable` or `demand`; `prod` works too but forces ≥10 nodes/job — see the
[queue table](#polaris-queues-alcf-docs)), use the launcher that submits server
job(s) + a dependent experiment job:

```bash
bash slurm/launch_servers_polaris.sh
```

1. Edit the `models` / `ngpus` arrays near the top (`"model_key:model_path"`;
   `ngpus` = tensor-parallel size).
2. Keep `config/examples/local_polaris.yaml` consistent: `fixed_interceptor`,
   `models[].model_key`, and `models[].model_id` (= the local path).
3. The launcher prints each server job id + port, the per-launch `SERVERS_DIR`, and
   the dependent experiment job id. The experiment job polls `ping_servers`, runs
   the games, then `qdel`s the servers.

- **How discovery works (no `qstat` parsing).** PBS job names can't carry
  `model_key:port`, so each server job writes its own
  `{model_key, model_id, urls, job_ids}` JSON — with its **HSN-routable IP**
  (`10.201.x.x`) — into a per-launch `SERVERS_DIR`. The experiment job points
  `DECRYPTO_SERVERS_FILE` at that directory; `get_available_servers()` merges the
  per-job files. (The Slurm `squeue` discovery path in `server.py` is untouched.)
- **✗ Run this on `debug` and it self-aborts.** The dependent `qsub` is rejected
  (`max_run=1`/`queued=1`); the launcher detects that, `qdel`s the orphan server so
  it doesn't run alone, and tells you to use `smoke_polaris.pbs`.

---

## Scaling to a bigger / more capable LLM

The 0.5B smoke proves the orchestration; real Decrypto play needs a 70B-class
model. The A100's 40 GB (vs Midway's 140 GB H200) is the binding constraint here.

- **Pre-stage the model first** (Step 2). A 70B/72B bf16 checkpoint is ~140 GB on
  disk — `huggingface-cli download … --local-dir $BASE/models/<name>` on a login
  node, before you submit anything.
- **Use tensor parallelism across the whole node.** 72B bf16 weights (~140 GB) do
  **not** fit one 40 GB A100. Serve at **TP=4** (the whole 4×A100 node = 160 GB) —
  e.g. `qsub -v MODEL_KEY=qwen2.5_72B,MODEL_PATH=$BASE/models/Qwen2.5-72B-Instruct,TP=4 slurm/smoke_polaris.pbs`.
  - **✗ Headroom is tight at TP=4 on 40 GB cards.** Weights alone eat ~35 GB/GPU,
    leaving little for the KV cache. If the engine OOMs on load or the KV cache is
    too small, lower `--max-model-len` (8192 → 4096) and/or nudge
    `--gpu-memory-utilization` up toward `0.95`. Both are currently hard-coded in
    `smoke_polaris.pbs` / `server_vllm_polaris.pbs` — edit them there.
  - **✗ A single A100 node may not be enough for the largest models** (or for long
    contexts at 70B). Multi-node tensor/pipeline parallel in vLLM needs Ray spanning
    nodes — that's a real bring-up, not a flag flip, and the current single-job
    smoke is one node only. Prefer fitting at TP=4 on one node; treat 2-node serving
    as a separate task.
- **Keep `model_key` / `model_id` consistent in three places** or discovery silently
  finds nothing: the `qsub -v MODEL_KEY/MODEL_PATH` (or launcher `models` array),
  and `config/examples/local_polaris.yaml`'s `fixed_interceptor` +
  `models[].model_key` + `models[].model_id` (the last must equal the served local
  path).
- **Consider dropping `--enforce-eager` once the model is stable** for a throughput
  win (it currently skips `torch.compile`). The cache dirs already live on eagle
  (`TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`), so the autotune write that killed
  the engine at Midway is mitigated — but re-verify on a probe before a long run.
- **Watch thread/process limits.** The per-job cgroup caps `pids.max = 4096`.
  `vllm serve` is light (one engine + one worker/GPU) so it fits, but if you see
  `pthread_create … Resource temporarily unavailable`, cap the BLAS/OMP thread
  family (`OMP_NUM_THREADS=8`, etc.).

---

## Running a longer session

The smoke is **1 episode, ≤1 h, one node, `debug`**. A real experiment needs more
episodes, a longer wall clock, and a queue that doesn't cap you at one job.

- **Move off `debug`.** Its 1-running/1-queued cap and ≤1 h walltime make it
  unusable for real runs. Pick from the [queue table](#polaris-queues-alcf-docs)
  by node count, walltime, and how many jobs you need running at once:
  - **For a long *single-node* run, use `capacity`** — 1–4 nodes, up to **168 h
    (7 days)**, and (unlike `preemptable`) it can't be preempted. It allows only
    **1 running job per user**, so stick to the **single-job** pattern
    (`smoke_polaris.pbs`-style, server + experiment fused) with the walltime raised.
  - **For a long *multi-node / two-job* run, use `preemptable`** — 1–10 nodes, up
    to **72 h**, and it allows ≥2 concurrent jobs (20 running/queued per project),
    which the server + dependent-experiment pattern needs.
    - **✗ `preemptable` jobs can be killed** when `demand`-queue work arrives. Add
      `#PBS -r y` so PBS reruns a preempted job, and make sure results are written
      incrementally so a kill mid-run isn't a total loss.
  - **✗ `prod` won't take a single-node job.** It's a routing queue with a
    **10-node-per-job minimum** (10–496 nodes), so requesting 1 node is rejected —
    only reach for it if you're genuinely running ≥10 nodes.
- **Raise the walltime in the `#PBS` headers.** The scripts ship with smoke-sized
  limits: `smoke_polaris.pbs` and `server_vllm_polaris.pbs` use
  `-l walltime=01:00:00`, `run_exp_polaris.pbs` uses `00:30:00`. Bump these to your
  queue's max for the experiment job, and keep the server's walltime **≥** the
  experiment's so the server doesn't die mid-run.
- **Raise the episode count in the config.** `config/examples/local_polaris.yaml`
  sets `num_episodes: 1` and `env_seed: [0]`. For a real run, increase
  `num_episodes` and widen `env_seed` to a list of seeds; set `verbose: false` to
  keep logs manageable.
- **Use the two-job launcher (Step 5), not the single-job smoke,** once you're on a
  multi-job queue: it puts the server and experiment on separate nodes (server GPUs
  busy, experiment node CPU/HTTP-only) and `qdel`s the server when the run ends, so
  the GPU is freed promptly instead of idling for the rest of a long walltime.
- **Bump the health/ready timeouts for big models.** A 70B load is far slower than
  the 0.5B's ~21 s. The server health wait (`HEALTH_TIMEOUT=900`) and the experiment
  ready-poll (`WAIT_TIMEOUT=1200`) are generous for the smoke but verify they cover
  your model's cold-load time, especially without `--enforce-eager`.
- **✗ Expect prologue / filesystem flakiness on long jobs.** Jobs can sit 20–40 min
  from `R` to script start, or hang the whole walltime with 0-byte output during an
  ALCF incident. If a job is silent, check `pbsnodes -l` for offlined nodes before
  assuming it's your bug — and `tail -f` the live `…wrap.log` to see real progress.

---

## Quick command reference

```bash
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/decrypto

# one-time (login node):
bash slurm/build_decrypto_serve_venv.sh            # build venv + tarball
huggingface-cli download <repo> --local-dir $BASE/models/<name>   # stage a model

# prove + smoke (debug queue, single node):
qsub slurm/probe_vllm_polaris.pbs                  # toolchain probe — run first
qsub slurm/smoke_polaris.pbs                       # server + 1 episode + results
qsub -v MODEL_KEY=...,MODEL_PATH=...,TP=4 slurm/smoke_polaris.pbs   # override model

# multi-node / multi-model (preemptable / demand, NOT debug):
bash slurm/launch_servers_polaris.sh

# watch / manage:
tail -f logs/paper/polaris_smoke_<jid>.log         # experiment progress
tail -f logs/vllm/<model_key>-<jid>.wrap.log       # server progress
qstat -u $USER          # my jobs        qdel <jid>     # cancel
qstat -Qf debug         # queue limits   pbsnodes -l    # offlined nodes
```
