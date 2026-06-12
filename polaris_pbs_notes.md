# Running Decrypto's self-hosted vLLM pipeline on ALCF Polaris (PBS)

Living document — the lab notebook for the **Decrypto self-hosted vLLM serving
pipeline** on ALCF **Polaris** (PBS Pro). It records the cluster facts, the port
from the RCC **Midway** (Slurm) version, the scale-up to three-model cross-play,
and every decision and job outcome along the way. The step-by-step recipe lives
in [`polaris_setup_guide.md`](polaris_setup_guide.md); the Slurm template and its
load-bearing fixes are in [`midway_notes.md`](midway_notes.md).

> **Scope:** self-hosted serving only — submit a job that runs `vllm serve` on
> Polaris A100 nodes, then a second job runs the Decrypto game loop against it
> over HTTP. This is **not** the ALCF inference-*gateway* path (Globus-authed
> HTTPS calls to models ALCF already hosts); that work lives on the
> `alcf-inference-gateway` branch (`midway_alcf_inference_notes.md`,
> `config/examples/argonne_polaris.yaml`, `src/utils/inference_auth_token.py`)
> and none of those files exist on this branch.
>
> The proven Polaris cluster facts and the gotcha log come from a sibling project
> on this same allocation: **`../MARSHAL/polaris_pbs_notes.md`** (path verified to
> resolve, 2026-06-12). MARSHAL is a *training* pipeline; only its
> **cluster-level** facts apply here (we serve/infer; no Ray/DeepSpeed/weight-sync).

---

## Status (2026-06-12)

- **Single-model pipeline: proven end-to-end.** The Qwen2.5-0.5B self-play smoke
  closed on a Polaris A100 (single-job 7186966, node `x3106c0s13b0n0`): venv
  staged from tarball (12 s), `vllm serve` healthy (21 s), discovery via
  `DECRYPTO_SERVERS_FILE` with the HSN IP, `ping_servers` healthy over
  `http://10.201.3.1:8159/v1` (0.55 s), 1/1 self-play episodes,
  `results/polaris_smoke/experiment_summary.csv` written, `Exit_status=0`
  (`logs/paper/polaris_smoke_7186966.log`). Gameplay quality at 0.5B is poor
  (JSON-format retries fail) — a model-capacity issue, not a pipeline issue,
  matching the Midway result.
- **Three-model scale-up: in progress.** Target experiment: full cross-play of
  **Qwen2.5-72B-Instruct + Qwen3-8B + Qwen3-4B** (27 encoder×decoder×interceptor
  combinations per env seed; config `config/examples/local_polaris_3model.yaml`).
  Qwen3-8B and Qwen3-4B are staged and probe Successful at TP=1 (job 7197265).
  The 70B-class slot was originally Llama-3.1-70B-Instruct; that repo is gated
  and no HF token exists on this machine (401 `GatedRepoError`, 2026-06-12), so
  the owner directed substituting the open Qwen2.5-72B-Instruct. Its TP=4 plan
  below is derived and awaiting on-cluster verification.

```bash
# Reproduce the proven single-model smoke (debug queue):
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/decrypto
qsub slurm/probe_vllm_polaris.pbs   # prove the toolchain first
qsub slurm/smoke_polaris.pbs        # server + 1 episode + results, one node
```

---

## Cluster facts (Polaris / ALCF)

Confirmed on this allocation (June 2026); cross-checked against MARSHAL's
bring-up and the [ALCF Polaris docs](https://docs.alcf.anl.gov/polaris/)
(re-verified 2026-06-12).

| Item | Value |
|---|---|
| Scheduler | **PBS Pro** (`qsub`/`qstat`/`qdel`), login hosts `polaris-login-01..04` |
| Account (`-A`) | **`lighthouse-uchicago`** — NOT "Uchicago-lighthouse" (PBS rejects that). Confirmed via `sbank-list-allocations`: alloc 12374, ~17,175 node-h available (2026-06-07) |
| GPU | **4× NVIDIA A100-SXM4-40GB** (sm_80, NVLink) per node. Re-confirmed 2026-06-12 on node `x3005c0s31b1n0` via `nvidia-smi` (40960 MiB/GPU, `logs/probe_nvidia-smi_7197265.txt`) and the [compute-nodes doc](https://docs.alcf.anl.gov/polaris/#polaris-compute-nodes) (160 GiB HBM2 per node). Polaris has no 80 GB A100 partition |
| CPU / RAM | AMD EPYC Milan 7543P, 64 hardware threads (`ncpus=64`), 512 GiB |
| Filesystems | `home`, `eagle`. Jobs **must** declare `-l filesystems=home:eagle` or PBS rejects them |
| Select line | `-l select=1:ncpus=64:ngpus=4` (do **not** add `:system=polaris`) |
| Node-local scratch | `/local/scratch` (RAID0 SSD) — fast + writable; used for `TMPDIR` + venv staging. Falls back to `/tmp` |
| Conda module | `module use /soft/modulefiles && module load conda/2025-09-25 && conda activate base` (venv built on top) |
| Native CUDA | `/soft/compilers/cudatoolkit/cuda-12.4.1`; driver 570.124.06 supports CUDA 12.8 → cu124 wheels are safe |
| Member base | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5` (repo, venv, models, caches) |
| Project root | `…/members/mehta5/decrypto` |
| Model store | `…/members/mehta5/models/` — see the staged-models table below |
| Serving venv | `…/members/mehta5/conda-envs/decrypto-serve` (+ tarball `…/members/mehta5/decrypto-serve-venv.tar`) |
| Caches | `…/members/mehta5/{hf_cache,triton_cache,torchinductor_cache}` |
| Job ids | `1234567.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`; bare tag via `${PBS_JOBID%%.*}` |
| Cross-node (HSN) | compute nodes `x3001c0s7b1n0`; HSN IPs `10.201.x.x`; FQDN `<node>.hsn.cm.polaris.alcf.anl.gov` |

### Queues (verified with `qstat -Qf <queue>` on 2026-06-12, cross-checked against the [running-jobs docs](https://docs.alcf.anl.gov/polaris/running-jobs/))

| Queue | Nodes/job | Walltime | Concurrency limits (`qstat -Qf`) | Fit for this pipeline |
|---|---|---|---|---|
| `debug` | 1–2 | 5 min – 1 h | `max_run=[u:1]`, `queued_jobs_threshold=[u:1]` | probes + single-job smokes only; cannot host the multi-job pattern |
| `debug-scaling` | 1–10 | 5 min – 1 h | 1 job/user | single-job only |
| `prod` (routing) | **10–496** | ≤ 24 h | 10 running/project | rejects 1-node jobs; not usable here |
| `preemptable` | 1–10 | ≤ **72 h** | `max_run=[p:10]`, `max_queued=[u:20]` | **the multi-job long-run queue** (3 servers + 1 experiment = 4 concurrent jobs). Jobs can be preempted by `demand` work without warning — submit with `-r y` and write results incrementally |
| `capacity` | 1–4 | ≤ 168 h | 1 running/user | single-job only; fallback for a no-preemption long run if servers + experiment are fused into one multi-node job (heavier script change; not implemented) |
| `demand` | 1–56 | ≤ 1 h | by request only | not applicable |

### PBS submit idiom (single GPU node)

```bash
qsub -A lighthouse-uchicago -q debug \
     -l select=1:ncpus=64:ngpus=4 -l filesystems=home:eagle \
     -l walltime=01:00:00 slurm/<script>.pbs
# (these are baked into each script's #PBS header; the launcher overrides
#  queue/walltime at submit time with `qsub -q ... -l walltime=... -r y`)
```

---

## Staged models and the tensor-parallel plan

Compute nodes are offline; every model is pre-downloaded on a login node into
`$BASE/models/` and served by **local path** (the config's `model_id` equals the
path, because that is the name vLLM serves under).

Per-GPU weight footprint ≈ `2 × params / TP` bytes (bf16). TP must divide both
`num_attention_heads` and `num_key_value_heads` (GQA) — head counts below were
read from each staged model's `config.json`, not from memory. KV-cache figures
are the vLLM startup log's own report, captured per job.

| Model (`model_key`) | HF repo | Disk | Heads (attn/KV) | TP | ≈ weights/GPU | Verification |
|---|---|---|---|---|---|---|
| `qwen3_8b` | `Qwen/Qwen3-8B` | 16 GB (5 shards) | 32 / 8 | 1 | ~16 GB | Successful — job 7197265: loaded at TP=1, mem_util 0.90, max_len 8192; KV cache 133,312 tokens (16.27× concurrency); generated (`logs/probe_wrap_7197265.log`) |
| `qwen3_4b` | `Qwen/Qwen3-4B` | 7.5 GB (3 shards) | 32 / 8 | 1 | ~8 GB | Successful — job 7197265: KV cache 189,648 tokens (23.15×); generated |
| `qwen2.5_72B` | `Qwen/Qwen2.5-72B-Instruct` (open) | 136 GiB (37 shards) | 64 / 8 (read from staged `config.json`; TP=4 → 16/2 per GPU) | **4** | 33.98 GiB measured | Successful — job 7197375: TP=4 is the only single-node fit on 40 GB cards, and it took three attempts to find the working engine config (see ledger 7197372/7197374): **mem_util 0.97 + `max_num_batched_tokens 2048` + `max_num_seqs 64`** → KV cache 18,800 tokens (2.29× concurrency @ 8192); generated. At mem_util ≤ 0.95 or default profiling settings the engine has zero KV memory |
| (dropped) `llama3.1_70B` | `meta-llama/Meta-Llama-3.1-70B-Instruct` (gated) | not staged | — | — | Unsuccessful — 401 `GatedRepoError` on a login node, 2026-06-12 (no HF token with an accepted Meta license on this machine). Owner substituted Qwen2.5-72B-Instruct rather than provide a token |
| `qwen2.5_0.5B` | `Qwen/Qwen2.5-0.5B-Instruct` | 954 MB | 14 / 2 | 1 | ~1 GB | Successful — probe 7186959 and smoke 7186966 (the original port) |

- Multi-node tensor/pipeline parallelism (Ray spanning nodes) is explicitly out
  of scope; 70B must fit one 4-GPU node at TP=4.
- Co-locating the two small models on one node (TP=1 each, `CUDA_VISIBLE_DEVICES`
  pinning + distinct ports) would save one node-job but is deferred until the
  one-model-per-node-job layout is proven; the current
  `server_vllm_polaris.pbs` runs one model per job.
- Disk check before the 70B pull: eagle reported 1.3 TB free (`df -h`,
  2026-06-12), comfortably above the ~141 GB needed.

---

## One-time setup

Done once per account, on a **login node** (has internet + HF reachability).

1. **Build the serving venv + tarball.** The stack mirrors MARSHAL's proven cu124
   set and adds the Decrypto runner deps, so **one env hosts both** the
   `vllm serve` server and the HTTP-only game runner:

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
   (compute nodes are **offline**):

   ```bash
   # on a login node, in the decrypto-serve venv:
   huggingface-cli download Qwen/Qwen3-8B --local-dir $BASE/models/Qwen3-8B
   huggingface-cli download Qwen/Qwen3-4B --local-dir $BASE/models/Qwen3-4B
   huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
     --local-dir $BASE/models/Qwen2.5-72B-Instruct
   # (gated repos like meta-llama/* additionally need an accepted license +
   #  HF_TOKEN on this machine; a tokenless attempt 401s immediately)
   ```

   After each download, verify completeness — a truncated pull is a classic
   silent failure: `config.json` present, `model.safetensors.index.json` present,
   and every shard in the index's `weight_map` exists on disk (verified for
   Qwen3-8B and Qwen3-4B on 2026-06-12).

---

## Running a smoke test

> **Which path to use — the `debug` queue forces a single job.** `debug`
> enforces **`max_run=1`** and **`queued_jobs_threshold=1`** per user
> (`qstat -Qf debug`). The faithful Midway-style **two-job** pattern (a server job
> + a dependent experiment job) therefore **cannot run on debug**: the two jobs
> can never be Running at once, and because the experiment `qdel`s the server they
> would deadlock. So:
> - **On `debug` (the smoke):** use the **single-job** `slurm/smoke_polaris.pbs`
>   (server + experiment in one job, one node).
> - **On a queue allowing ≥2 concurrent jobs** (`preemptable`): use the
>   **two-job** `slurm/launch_servers_polaris.sh` (multi-node, multi-model).

**Always run the probe first:** `qsub slurm/probe_vllm_polaris.pbs` proves the
toolchain (nvidia-smi → torch sees the A100 → vLLM loads and generates). The
probe now accepts a multi-model spec, e.g.
`qsub -v MODEL_SPECS="$BASE/models/Qwen3-8B:1:0.90:8192;$BASE/models/Qwen3-4B:1:0.90:8192" slurm/probe_vllm_polaris.pbs`
(entries are `path:tp:mem_util:max_len`, semicolon-separated because `qsub -v`
splits on commas). Do not run a smoke until the probe passes.

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

### Two-job launcher (preemptable, multi-node, multi-model)

1. The `models` / `ngpus` / `mem_utils` / `max_lens` arrays at the top of
   `slurm/launch_servers_polaris.sh` define the served set — currently the three
   cross-play models at TP 4/1/1. Queue, walltimes, config name, seeds and
   episode count are env knobs (`QUEUE`, `SERVER_WALLTIME`, `EXP_WALLTIME`,
   `CONFIG_NAME`, `EXP_NAME`, `SEEDS`, `NUM_EPISODES`, `HEALTH_TIMEOUT`,
   `WAIT_TIMEOUT`).
2. `config/examples/local_polaris_3model.yaml` must stay consistent:
   `models[].model_key` matches what each server job registers, and
   `models[].model_id` equals the served **local path**.
3. **Submit** `bash slurm/launch_servers_polaris.sh` from the repo root. It
   prints each server job id + port, the per-launch `SERVERS_DIR`, and the
   dependent experiment job id; the experiment job waits for all
   `EXPECTED_MODELS` servers to answer `ping_servers`, runs the games, writes
   `results/<EXP_NAME>/experiment_summary.csv`, and `qdel`s the servers.
   (On debug the dependent submit is rejected; the launcher then `qdel`s its own
   server jobs and points you at `smoke_polaris.pbs`.)

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
  This includes the legacy `agent_paths` table in `src/utils/server.py`
  (2026-06-12): the staged keys (`llama3.1_70B`, `qwen3_8b`, `qwen3_4b`,
  `qwen2.5_0.5B`) now point at `$BASE/models/...`; that table is read only by the
  Slurm `squeue` branch (on Polaris, `model_id` travels inside each server's
  discovery JSON), but stale Midway paths were fixed so no dead pointers remain.
- **Offline + caches.** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, `HF_HOME` on
  eagle, serve a **local model path** — compute nodes have no internet.
- **GPU memory.** A100 40 GB vs H200 140 GB: the 0.5B proved the pipeline at
  TP=1; the 70B-class target requires TP=4 across the whole node (see the
  tensor-parallel plan above).

---

## Deliverables (this port)

| File | Role |
|---|---|
| `slurm/build_decrypto_serve_venv.sh` | one-time: build the serving venv + tarball on a login node |
| `slurm/probe_vllm_polaris.pbs` | toolchain probe; multi-model `MODEL_SPECS` (`path:tp:mem:len;…`) |
| `slurm/smoke_polaris.pbs` | **single-job** smoke (server + experiment, one node) — the `debug`-queue path |
| `slurm/server_vllm_polaris.pbs` | per-model vLLM server job; registers itself via `DECRYPTO_SERVERS_FILE`; `GPU_MEM_UTIL`/`MAX_MODEL_LEN`/`HEALTH_TIMEOUT` parameterized |
| `slurm/launch_servers_polaris.sh` | two-job `qsub` launcher: N server jobs + dependent experiment job (preemptable) |
| `slurm/run_exp_polaris.pbs` | dependent experiment job: poll `ping_servers`, run `run.py`, `qdel` servers; `CONFIG_NAME`/`EXP_NAME`/`SEEDS`/`NUM_EPISODES` parameterized |
| `config/examples/local_polaris.yaml` | 0.5B self-play smoke config (proven; do not overwrite) |
| `config/examples/local_polaris_3model.yaml` | 3-model cross-play config: 27 combos/seed, Qwen3 thinking budgets |
| `src/utils/server.py` | `DECRYPTO_SERVERS_FILE` may be a dir of per-job JSON files (Slurm path intact); `agent_paths` localized to Polaris |

---

## Carry-over gotchas (from Midway / MARSHAL — they apply here)

- **`--enforce-eager` for the smoke** — skips `torch.compile`, avoiding the
  autotune-cache write that killed the engine on load at Midway.
- **Scrub inherited `TMPDIR`** → set it under `/local/scratch/$USER_<jobtag>`.
- **Do not `pip install -r requirements.txt`** — it pins torch 2.9 / vllm 0.13
  (unvalidated here). Use the proven cu124 set in `build_decrypto_serve_venv.sh`.
- **`transformers<5` + `tokenizers<0.22`** — a fresh install pulls transformers 5,
  which dropped `all_special_tokens_extended` that vLLM 0.8.x calls.
- **per-job cgroup `pids.max = 4096`** — bit MARSHAL's many-process Ray training
  hard, but `vllm serve` is light (one engine + one worker/GPU) and should fit. If
  you see `pthread_create … Resource temporarily unavailable`, cap the BLAS/OMP
  thread family (`OMP_NUM_THREADS=8`, etc.). The same limit exists on login nodes
  (`git grep` without `--threads=1` failed with the same error, 2026-06-12).
- **Polaris prologue/filesystem flakiness** — jobs can sit 20–40 min from `R` to
  script start, or hang the whole walltime with 0-byte output during an ALCF
  incident. If a job is silent, check `pbsnodes -l` for offlined nodes before
  assuming your bug. Every script streams a live `…wrap.log` so you can watch.
- **The ALCF proxy hangs in-cluster HTTP** (bit us at smoke 7186962). Polaris
  **compute** nodes export `http_proxy=http://proxy.alcf.anl.gov:3128`
  (via `/etc/sysconfig/proxy`) for outbound internet. `curl` AND the OpenAI/httpx
  client honor it, so a request to `localhost:<port>` or an HSN `10.201.x` address
  gets routed through the (air-gapped, unreachable) proxy and **hangs** — vLLM was
  up with `Application startup complete`, yet the health check timed out after
  900 s. **Fix:** in every job `unset http_proxy https_proxy HTTP_PROXY
  HTTPS_PROXY all_proxy …` and set `no_proxy` to cover `localhost,127.0.0.1,
  10.201.0.0/16,.hsn.cm.polaris.alcf.anl.gov` (we never need outbound — fully
  offline). Health checks also use `curl --noproxy '*'` + a `urllib`
  ProxyHandler({}) fallback, and hit `127.0.0.1` (not `localhost`, to dodge an
  IPv6 `::1` detour).
- **Qwen3 models think before answering.** Qwen3-8B/-4B emit `<think>` blocks; if
  `max_tokens` is too small the model spends its whole budget thinking and never
  produces the JSON the runner parses — format-retry failures that look like a
  pipeline bug but are a token-budget issue. `LocalModel.max_reasoning_tokens`
  (two-stage pattern in `src/agents/role_client.py`: thinking call capped at the
  budget with `stop=</think>`, then an answer call that prefills `</think>`)
  bounds this; the 3-model config sets `max_tokens: 2500`,
  `max_reasoning_tokens: 2000`, following `config/paper/figure_4_tom_piaget.yaml`.

---

## Job ledger

Every log file under `logs/` mapped to its job and outcome (`logs/` is
gitignored — enumerate with `ls logs/` / depth-limited `find logs -maxdepth 2`,
never a cluster-wide `find`).

| Log file(s) | Job id | Queue | Model / TP | Outcome | Root cause / note |
|---|---|---|---|---|---|
| `logs/probe_wrap_7186959.log`, `logs/probe_nvidia-smi_7186959.txt`, `logs/7186959.*.OU/.ER` | 7186959 | debug | Qwen2.5-0.5B / TP1 | Successful | Toolchain probe: torch saw the A100, vLLM 0.8.4 loaded the 0.5B on V1 and generated; `Exit_status=0` |
| `logs/vllm/qwen2.5_0.5B-7186960.wrap.log`, `logs/vllm/7186960.*.OU/.ER` | 7186960 | debug | Qwen2.5-0.5B / TP1 | Unsuccessful (aborted) | Orphaned two-job server; `qdel`'d because the dependent experiment job was rejected on debug (`max_run=1`/`queued=1`). Demonstrates the two-job pattern is invalid on debug |
| `logs/vllm/qwen2.5_0.5B-7186962.out`, `logs/paper/polaris_smoke_7186962.log`, `logs/paper/7186962.*.OU/.ER` | 7186962 | debug | Qwen2.5-0.5B / TP1 | Unsuccessful | vLLM came up (`Application startup complete`) but the health check hit the ALCF `http_proxy` and hung → `vLLM not healthy in 900s` (`Exit_status=45`). Fixed by the proxy bypass |
| `logs/vllm/qwen2.5_0.5B-7186966.out`, `logs/paper/polaris_smoke_7186966.log`, `logs/paper/7186966.*.OU/.ER` | 7186966 | debug | Qwen2.5-0.5B / TP1 | Successful | Full single-job orchestration: health in 21 s, discovery via HSN IP `10.201.3.1:8159`, `ping_servers` 0.55 s, 1 episode, `experiment_summary.csv` written; `Exit_status=0`. (Gameplay garbage = 0.5B model capacity, not pipeline) |
| `logs/build/decrypto_serve_venv.log`, `logs/build/runner_import_check.log`, `logs/build/vllm_help_check.log` | n/a (login) | — | — | Successful | One-time serving-venv build + import/`vllm --help` sanity checks |
| `logs/build/download_qwen3_8b.log`, `logs/build/download_qwen3_4b.log` | n/a (login) | — | Qwen3-8B, Qwen3-4B | Successful | Login-node `huggingface-cli download` (2026-06-12, ~3 min each); all shards + `model.safetensors.index.json` verified present |
| `logs/probe_wrap_7197265.log`, `logs/probe_nvidia-smi_7197265.txt`, `logs/7197265.*.OU/.ER` | 7197265 | debug | Qwen3-8B / TP1 + Qwen3-4B / TP1 | Successful | Multi-model probe on `x3005c0s31b1n0` (~77 s): A100-SXM4-40GB confirmed (40960 MiB); Qwen3-8B KV cache 133,312 tokens (16.27× @ 8192), Qwen3-4B KV cache 189,648 tokens (23.15×); both generated; `Exit_status=0` |
| `logs/build/download_qwen2.5_72b.log` | n/a (login) | — | Qwen2.5-72B | Successful | Login-node download (~25 min, 136 GiB); 37/37 shards verified against `model.safetensors.index.json`; `config.json` reads 64 attn / 8 KV heads / 80 layers |
| `logs/probe_wrap_7197372.log`, `logs/7197372.*.OU/.ER` | 7197372 | debug | Qwen2.5-72B / TP4 | Unsuccessful | Weights loaded (33.98 GiB/GPU in 273 s — TP=4 fits) but at mem_util 0.95 / max_len 8192 the engine's profiling pass left no memory for KV blocks: `ValueError: No available memory for the cache blocks` → worker SIGKILL (rc=137). First fix attempt: cap the profiling/prefill batch |
| `logs/probe_wrap_7197374.log`, `logs/7197374.*.OU/.ER` | 7197374 | debug | Qwen2.5-72B / TP4 | Unsuccessful | Same `No available memory for the cache blocks` despite `max_num_batched_tokens=2048` being active (`Chunked prefill is enabled with max_num_batched_tokens=2048` in the log) — the profiling peak is dominated by fixed costs (non-torch NCCL/IPC buffers for TP=4 + the dummy-sampler logits, which scale with `max_num_seqs`, default 1024, ~150k vocab), not the prefill batch. Next: mem_util 0.97 + `max_num_seqs 64` (job 7197375) |
| `logs/probe_wrap_7197375.log`, `logs/probe_nvidia-smi_7197375.txt`, `logs/7197375.*.OU/.ER` | 7197375 | debug | Qwen2.5-72B / TP4 | Successful | On `x3204c0s7b0n0`: TP=4, mem_util 0.97, max_len 8192, `max_num_batched_tokens 2048`, `max_num_seqs 64`. Weights 33.98 GiB/GPU in 232 s; **KV cache 18,800 tokens (2.29× concurrency at 8,192 tokens/request)**; generated a completion; `Exit_status=0`. This is the verified 72B serving configuration |
| `logs/paper/polaris_smoke_7197376.log`, `logs/vllm/qwen2.5_72B-7197376.out`, `logs/paper/7197376.*.OU/.ER` | 7197376 | debug | Qwen2.5-72B / TP4 | Successful | Fused server-sanity smoke (RUN_EPISODE=0) on `x3204c0s7b0n0`: `vllm serve` healthy in 75 s (warm node — same node as 7197375; cold load is ~240 s), registered `http://10.201.0.184:8615/v1` via `DECRYPTO_SERVERS_FILE`, `ping_servers` replied in 1.45 s, direct HTTP chat completion in 1.0 s, KV cache 18,800 tokens re-confirmed over the serve path; "SMOKE (server-sanity mode) PASSED" |
| `logs/vllm/qwen3_8b-7197358.wrap.log`, `logs/vllm/7197358.*.OU/.ER` | 7197358 | preemptable | Qwen3-8B / TP1 | Partially successful | 2-model mechanics test server: preempted once and requeued by `-r y` (run_count=2), came healthy, registered, answered `ping_servers` (200 OK in its log); `qdel`'d by the experiment's failure path at 12:44 |
| `logs/vllm/qwen3_4b-7197359.wrap.log`, `logs/vllm/7197359.*.OU/.ER` | 7197359 | preemptable | Qwen3-4B / TP1 | Unsuccessful | **Died at exactly its walltime while idle-serving** (stime 08:11:38 → obittime 09:42:25 = 1:30:47 ≈ walltime 01:30, `Exit_status=-29`, run_count=2): the dependent experiment never got a node while this server burned its clock. The queue-skew failure mode |
| `logs/paper/polaris_smoke_7197360.log`, `logs/paper/7197360.*.OU/.ER` | 7197360 | preemptable | 2-model experiment | Unsuccessful | Ran (and was itself preempted/rerun) three times; in the final run only 1/2 servers still existed (the 4B had hit walltime) → `FATAL: only 1/2 servers ready after 1800s`, exit 46, correctly `qdel`'d the surviving server. Validated: dependency release, ready-poll, SEEDS passthrough (visible in env), failure-path qdel |
| `logs/vllm/qwen3_8b-7197380.wrap.log`, `logs/vllm/7197380.*.OU/.ER` | 7197380 | preemptable | Qwen3-8B / TP1 | Unsuccessful | 3-model launch server: healthy in 37 s, registered `10.201.4.87:8421` — then died at exactly its walltime (stime 10:04:05 → obittime 12:35:10 ≈ 02:30, `Exit_status=-29`) while the experiment job was still queued. Cleanup removed its JSON |
| `logs/vllm/qwen3_4b-7197381.wrap.log`, `logs/vllm/7197381.*.OU/.ER` | 7197381 | preemptable | Qwen3-4B / TP1 | Unsuccessful | Same walltime-skew death: healthy in 27 s, registered `10.201.4.160:8807`, died 12:13:27 after its full 02:30 walltime, experiment still queued |
| `logs/vllm/qwen2.5_72B-7197379.wrap.log`, `logs/vllm/7197379.*.OU/.ER` | 7197379 | preemptable | Qwen2.5-72B / TP4 | Partially successful | Started 12:44 (3.4 h queue skew after submission at 05:21), healthy, registered `7197379.json` at 12:49, answered the experiment's pings (the "1/3 ready") for an hour — then `qdel`'d by 7197382's failure path at 13:45. The 72B serving config itself worked over the two-job path |
| `logs/paper/polaris_smoke_7197382.log`, `logs/paper/7197382.*.OU/.ER` | 7197382 | preemptable | 3-model experiment | Unsuccessful | Started 12:44 alongside the 72B, but the two Qwen3 servers had already died at their walltimes hours earlier; replacement servers (7197525/7197526, submitted 12:48 into the same SERVERS_DIR) were still queued when WAIT_TIMEOUT expired: `FATAL: only 1/3 servers ready after 3600s`, exit 46, qdel'd the surviving 72B. 7197525 started at almost that exact minute; both replacements were then qdel'd as orphans |
| (no log — qdel'd while queued) | 7197525, 7197526, 7197528 | preemptable | replacements + fused copy | Aborted | 7197525/26: orphaned Qwen3 replacement servers, qdel'd after their experiment died. 7197528: preemptable copy of the fused smoke, qdel'd once the capacity copy started first |
| `logs/paper/polaris_fused_7197574.log`, `logs/vllm/{qwen2.5_72B,qwen3_8b,qwen3_4b}-7197574.wrap.log`, `logs/paper/7197574.*.OU/.ER` | 7197574 | **capacity** | 3 servers + experiment, fused 4-node job | **Successful — rungs 3+4** | Started 1 min after submission. All 3 servers ready in **361 s** (one per node via `mpiexec --hosts`); ran the full **27-combination** cross-play matrix (1 seed × 1 episode); `run.py` rc=0; `results/polaris_3model_fused_cap/experiment_summary.csv` = 27 rows, verified to contain all 27 unique (encoder, decoder, interceptor) triples; per-combo dirs written incrementally. Total job 25 min. "FUSED RUN COMPLETE" |

---

## Decisions / changes log

- **2026-06-07 — Orientation.** Confirmed Polaris login (PBS Pro). Verified
  account **`lighthouse-uchicago`** (`sbank`: alloc 12374, ~17,175 node-h),
  `debug` queue present. Confirmed the **self-hosted serving** task (NOT the
  Globus inference gateway). Read the Midway trio + `src/utils/server.py` and
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
- **2026-06-07 — Probe Successful (7186959, node x3106c0s19b1n0).** ~80 min queue
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
- **2026-06-07 — Blocker: the two-job dependent pattern can't run on `debug`.**
  `bash slurm/launch_servers_polaris.sh` submitted the server (7186960) but
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
  (7186962, debug).
- **2026-06-07 — Smoke 7186962 Unsuccessful: ALCF proxy hangs the health check
  (not a vLLM problem).** Started immediately on x3106c0s19b1n0; venv staged in
  9 s, but `vLLM not healthy in 900s` (Exit_status 45). The vLLM server log showed
  it came up cleanly in ~30 s (`Application startup complete`, API on
  `http://0.0.0.0:8305`, `/health` route live) — so the **health check** was
  wrong, not vLLM. Root cause: compute nodes set
  `http_proxy=proxy.alcf.anl.gov:3128`, so `curl localhost:8305` routed through
  the unreachable proxy. **Fix:** unset all proxy vars + set `no_proxy` in every
  job; proxy-immune health check (`curl --noproxy '*'` + urllib ProxyHandler({}),
  hitting 127.0.0.1) with a one-time proxy/socket diagnostic. Resubmitted as
  **7186966**.
- **2026-06-07 — Smoke Successful (7186966, node x3106c0s13b0n0).** With the proxy
  bypass: venv staged in 12 s, vLLM healthy in **21 s**, server registered with its
  HSN IP `http://10.201.3.1:8159/v1`, `ping_servers` got a healthy reply in 0.55 s
  (file-based discovery + real HTTP over the HSN NIC), `run.py +get_models_from_slurm=true`
  discovered the server and completed **1/1** episodes, wrote
  `results/polaris_smoke/experiment_summary.csv`, and the cleanup trap stopped
  vLLM. `Exit_status=0`. The 0.5B model's gameplay is poor (JSON-retry failures) —
  a model-capacity issue, not a pipeline one, matching Midway. **The self-hosted
  vLLM serving pipeline is proven on Polaris.**
- **2026-06-12 — Began the 3-model cross-play scale-up** (Llama-3.1-70B +
  Qwen3-8B + Qwen3-4B, 27 combos/seed). Re-verified cluster facts: `qstat -Qf`
  for `debug` (max_run 1/user) and `preemptable` (1–10 nodes, 72 h,
  max_run 10/project, max_queued 20/user) match the ALCF running-jobs docs;
  compute-nodes doc + on-node `nvidia-smi` (job 7197265) confirm 4× A100 40 GB —
  no 80 GB partition, so the 70B TP plan is TP=4. Confirmed
  `../MARSHAL/polaris_pbs_notes.md` resolves.
- **2026-06-12 — Staged Qwen3-8B (16 GB) and Qwen3-4B (7.5 GB)** on a login node
  (`logs/build/download_qwen3_{8b,4b}.log`); verified `config.json` + all shards
  against each `model.safetensors.index.json`. **Llama-3.1-70B staging blocked:**
  the repo `meta-llama/Meta-Llama-3.1-70B-Instruct` is gated and no HF token
  exists on this machine (`huggingface-cli whoami` → "Not logged in";
  download attempt → 401 `GatedRepoError`, recorded).
- **2026-06-12 — 70B-class slot substituted: Qwen2.5-72B-Instruct.** Presented
  the owner three options (provide an HF token for the official gated repo; use
  the ungated NousResearch mirror; substitute the open Qwen2.5-72B-Instruct).
  The owner chose **Qwen2.5-72B-Instruct** — open repo, no token, and it was the
  original scaling target named in this notebook. Consequences: `model_key` is
  `qwen2.5_72B` everywhere (config, launcher, `agent_paths`); bf16 weights are
  ~145 GB (slightly heavier than Llama-70B's ~141 GB), so the per-GPU margin at
  TP=4 shrinks — the launcher sets `--gpu-memory-utilization 0.95` for the 72B
  (at 0.92 the ~0.4 GB/GPU KV headroom would not cover an 8192-token max_len).
  Download started (`logs/build/download_qwen2.5_72b.log`); the empty
  `Meta-Llama-3.1-70B-Instruct` dir skeleton from the 401 attempt was removed.
- **2026-06-12 — 72B probe attempt 1 (7197372) Unsuccessful: zero KV memory at
  TP=4 / mem_util 0.95 / max_len 8192.** The load itself succeeded — vLLM
  reported `Model loading took 33.9835 GiB` per GPU in 273 s, confirming the
  TP=4 weight fit on 40 GB cards — but `_initialize_kv_caches` raised
  `No available memory for the cache blocks` and the worker died with rc=137.
  Arithmetic: budget 0.95 × 40 GiB = 38 GiB; weights 33.98 GiB; the remaining
  ~4 GiB was consumed by the memory-profiling forward pass at the default
  8192-token batch. Chosen fix (over halving `max_model_len`, which would risk
  truncating late-game Decrypto prompts): cap `--max-num-batched-tokens` at
  2048 so the profiling/prefill activation peak shrinks; chunked prefill (on by
  default in the V1 engine) prefills longer prompts in 2048-token chunks.
  Expected KV ≈ 2.5 GiB/GPU ≈ 32k tokens at 80 KiB/token-per-GPU (80 layers ×
  2-of-8 KV heads × 128 dim × K+V × bf16). Added the knob to probe/server/smoke
  scripts + a per-model `mnbts` array in the launcher; resubmitted as 7197374.
- **2026-06-12 — 72B probe attempt 2 (7197374) Unsuccessful, attempt 3 (7197375)
  Successful — the verified 72B config.** Attempt 2 kept mem_util 0.95 and
  added `max_num_batched_tokens=2048`; the engine still reported
  `No available memory for the cache blocks`, proving the profiling peak is
  dominated by fixed costs, not the prefill batch: non-torch allocations (NCCL +
  IPC buffers for TP=4, CUDA context) plus the profiling dummy-sampler whose
  logits tensors scale with `max_num_seqs` (default 1024 × ~152k vocab).
  Attempt 3 raised mem_util to **0.97** and capped **`max_num_seqs` to 64**
  (well above the ~27 concurrent games the cross-play generates): KV cache
  **18,800 tokens**, 2.29× concurrency at 8,192-token requests, generation
  Successful (`logs/probe_wrap_7197375.log`, node x3204c0s7b0n0). Real Decrypto
  prompts are far shorter than 8,192 tokens, so effective concurrency is
  higher; requests beyond KV capacity queue inside vLLM. All serving scripts
  now expose `MAX_NUM_BATCHED_TOKENS` / `MAX_NUM_SEQS`, and the launcher pins
  the 72B to `0.97 / 8192 / 2048 / 64`.
- **2026-06-12 — Rung 2 Successful: fused 72B server smoke (7197376).** Added a
  `RUN_EPISODE=0` server-sanity mode to `smoke_polaris.pbs` (health + discovery
  registration + `ping_servers` + one direct HTTP chat completion, skipping the
  run.py episode — the 0.5B-specific `local_polaris` config does not apply to a
  72B-only smoke, and an episode may not fit debug's 1 h walltime). With the
  verified 72B config: healthy in 75 s (warm node; cold ~240 s), HSN
  registration and a 1.0 s completion over `/v1/chat/completions`. The full
  `vllm serve` HTTP path for the 72B is proven.
- **2026-06-12 — Launched the 3-model orchestration smoke (rungs 3+4) on
  `preemptable`:** servers 7197379 (qwen2.5_72B), 7197380 (qwen3_8b), 7197381
  (qwen3_4b) + dependent experiment 7197382 (`local_polaris_3model`, 1 seed ×
  1 episode = 27 games, `exp_name=polaris_3model`), server walltime 02:30,
  experiment 02:00. The experiment log's ready-poll (3/3 servers) is the rung-3
  evidence; the 27-combo `experiment_summary.csv` is rung 4's.
- **2026-06-12 — Launcher-mechanics smoke on `preemptable` (first multi-job
  launch).** Before the 72B is staged, validated the multi-server machinery with
  the two Qwen3 models alone: added an `ONLY_MODELS` subset filter to
  `launch_servers_polaris.sh` and a `local_polaris_2model_smoke.yaml` config
  (2×2×2 = 8 combos × 2 seeds = 16 games; also exercises the space-separated
  `SEEDS` passthrough). Submitted servers 7197358 (qwen3_8b) + 7197359
  (qwen3_4b) and dependent experiment 7197360. Outcome recorded in the ledger
  when complete.
- **2026-06-12 — The two-job pattern's real failure mode on a contended queue:
  walltime skew (jobs 7197359/7197380/7197381 Unsuccessful).** All three
  servers died with `Exit_status=-29` at *exactly* their requested walltime
  (obittime − stime = walltime to within a minute), having served idle the
  whole time: PBS schedules the server jobs and the dependent experiment job
  independently, and on a contended `preemptable` queue (55 queued / 38 running
  observed) the experiment's own node never freed before the servers' clocks
  ran out. `-W depend=after:` only orders *starts*; it provides no
  co-scheduling. Two further observations from the same window: preemption
  with `-r y` does requeue (7197358, run_count=2), and a rerun server
  re-registers at a *new* address — which the runner, reading URLs once at
  startup, would never see; a mid-run preemption of any server therefore
  breaks a long run even if the server itself recovers. Conclusions: (1)
  two-job runs need server walltime ≫ experiment walltime + worst-case queue
  skew, and (2) the robust vehicle for the 8–16 h production run is a **fused
  single multi-node job** — one queue wait, atomic lifetime, and eligible for
  `capacity` (1–4 nodes, ≤168 h, **no preemption**, 1 running job/user).
- **2026-06-12 — Tactical salvage + fused job authored.** The 3-model
  experiment (7197382) and the 72B server (7197379) finally started ~12:44 with
  the two Qwen3 servers already dead; resubmitted them (7197525 qwen3_8b,
  7197526 qwen3_4b, walltime 03:00) into the *same* `SERVERS_DIR` — discovery
  merges by `model_key`, and the dead servers' cleanup traps had removed their
  stale JSONs, so the experiment simply waits for the newcomers (its
  WAIT_TIMEOUT runs to ~13:45). In parallel, authored the structural fix:
  `slurm/fused_3model_polaris.pbs` (one 4-node job: node 0 experiment, nodes
  1–3 one `vllm serve` each via `mpiexec --hosts`, using the extracted
  per-node bootstrap `slurm/start_vllm_server_node.sh`) and submitted it as a
  27-game smoke (7197528, `exp_name=polaris_3model_fused`). Whichever attempt
  completes first closes rungs 3+4; the fused path is the production vehicle.
- **2026-06-12 — Two-job attempt abandoned after a second skew loss.** 7197382
  timed out (1/3 ready after 3600 s) at 13:45 — the minute its replacement 8B
  server finally started — and its failure path `qdel`'d the healthy 72B
  (7197379). Cut losses: `qdel`'d the now-orphaned replacements 7197525/7197526
  and consolidated on the fused single-job design. Submitted a second copy of
  the fused smoke to **`capacity`** (7197574, `exp_name=polaris_3model_fused_cap`;
  the queue showed 9 running / 6 queued, 1 running job per project, ≤4 nodes,
  no preemption) alongside the preemptable copy (7197528) — whichever starts
  first closes the rung, the other gets `qdel`'d, and capacity is the planned
  home for the production run regardless.
- **2026-06-12 — Rungs 3+4 Successful: fused 27-combo smoke (7197574,
  capacity).** The capacity copy started **one minute** after submission
  (versus 3.4–7 h queue skews on preemptable all day). All three servers came
  up on their own nodes via `mpiexec --hosts` and answered `ping_servers` in
  **361 s** (rung 3: 3/3 ready, three `<jobtag>_<model_key>.json` files in the
  fused `SERVERS_DIR`); the experiment then ran **all 27 role combinations**
  (1 seed × 1 episode), `run.py` rc=0, and
  `results/polaris_3model_fused_cap/experiment_summary.csv` holds exactly the
  27 unique (encoder, decoder, interceptor) triples (verified
  programmatically) with per-combo game dirs written incrementally as games
  finished (rung 4). Whole job: 25 min. The preemptable duplicate (7197528)
  was qdel'd unrun. The two-stage Qwen3 thinking pattern is visible working in
  the server logs (prefilled `</think>` + answer call). Production sizing
  note: 27 games completed in ~19 min of game time; the runner spawns one
  process per game, so the production seed count is bounded by the experiment
  node's `pids.max=4096` — added `OMP_NUM_THREADS=1` to the fused experiment
  node and chose 15 seeds (405 games / 405 processes) for rung 5.
- **2026-06-12 — Derived the TP plan from staged `config.json` files** (see the
  staged-models table): Qwen3-8B/-4B have 32 attention / 8 KV heads → TP=1
  (single 40 GB card holds 16/8 GB of weights with ample KV headroom);
  Llama-3.1-70B (64/8 heads expected) only fits a node at TP=4 (~35 GB
  weights/GPU). Mitigation ladder for the expected-tight 70B KV cache:
  `--max-model-len` 8192 → 4096 → 2048 and `--gpu-memory-utilization` up to 0.95.
- **2026-06-12 — Multi-model probe Successful (7197265, x3005c0s31b1n0).**
  Extended `probe_vllm_polaris.pbs` to take `MODEL_SPECS` (`path:tp:mem:len`,
  semicolon-separated — `qsub -v` splits on commas, which cost one observation:
  list-valued vars must avoid `,`). Qwen3-8B then Qwen3-4B each loaded at TP=1
  with mem_util 0.90 / max_len 8192 and generated; KV caches 133,312 and 189,648
  tokens respectively; whole job ~77 s including venv staging; `Exit_status=0`.
- **2026-06-12 — Parameterized the serving scripts for the 3-model run** (commit
  `31ce823`): per-model `GPU_MEM_UTIL`/`MAX_MODEL_LEN`/`HEALTH_TIMEOUT` on the
  server job; `CONFIG_NAME`/`EXP_NAME`/`WAIT_TIMEOUT`/`SEEDS`/`NUM_EPISODES` on
  the experiment job; launcher targets `preemptable` with `-r y` and
  walltime flags at submit time (headers stay debug-sized defaults). Authored
  `config/examples/local_polaris_3model.yaml`. Fixed the stale Midway
  `agent_paths` in `src/utils/server.py` (legacy `squeue`-only table; dir-merge
  loader re-verified with an ad-hoc unit test).
