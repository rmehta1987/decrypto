# Handoff: port Decrypto's self-hosted vLLM pipeline to ALCF Polaris (PBS)

> Paste everything below the line into a fresh Claude Code session running on a
> Polaris login node, inside a checkout of this repo. It is written **to that
> Claude**. It was authored from the RCC Midway checkout (branch
> `midway-decrypto`), where the Slurm version of this pipeline is already
> working end-to-end.
>
> **v2 addendum (2026-06):** since this was authored, a sibling project (MARSHAL, a
> ROLL training pipeline) was brought up GREEN on the SAME Polaris allocation. The
> Polaris facts this prompt told you to "go confirm" are now **confirmed**, and several
> gotchas are known in advance. Those are collected in a new section,
> **"✅ CONFIRMED Polaris facts + proven gotchas,"** right after "Read these first."
> The proven blow-by-blow record is at **`../MARSHAL/polaris_pbs_notes.md`**. Nothing
> else below was changed — the new section just supersedes the later
> "Polaris facts to CONFIRM" section with verified values.

---

You are a Claude Code session on **ALCF Polaris**. Your job is to get the
**Decrypto self-hosted vLLM serving pipeline** running on Polaris and prove it
with a one-episode self-play smoke test on Polaris GPUs.

Polaris uses **PBS Pro** (`qsub`/`qstat`/`qdel`), not Slurm. This same pipeline
was already ported to **UChicago RCC Midway (Slurm)**, and that port is your
template. Your task is the PBS analog of that Midway port: scheduler scripts,
the squeue-based server discovery, account/allocation info, and storage paths.

## ⚠️ Do not conflate the two "Polaris" paths in this repo

There are two completely different ways this repo can use ALCF, and they are
easy to mix up because both say "Polaris":

1. **ALCF inference *gateway*** — `midway_alcf_inference_notes.md`, `src/utils/inference_auth_token.py`,
   `src/utils/globus_login.py`, `src/utils/list_argonne_endpoints.py`,
   `config/examples/argonne_polaris.yaml`, and the `use_globus_auth` field on
   `LocalModel`. This makes **HTTPS calls to models ALCF already hosts** — no
   local GPU, no scheduler. It is currently blocked on Globus group access.
   **This is NOT your task. Do not modify or rely on it.**

2. **Self-hosted vLLM serving on Polaris compute nodes via PBS** — you submit a
   job that runs `vllm serve` on Polaris A100s, then a second job runs the
   Decrypto game loop against that server over HTTP. **This is your task.** It is
   the direct analog of the Midway Slurm scripts.

If you find yourself touching anything Globus/`inference_auth_token`-related,
stop — you're in the wrong path.

## Read these first (the Midway template)

- `midway_notes.md` — the step-by-step guide + decisions log for the Slurm port.
  **Your deliverable is the PBS equivalent of this document.** Mirror its
  structure (cluster facts table, one-time setup, smoke test, what-changed,
  results, decisions log).
- `slurm/launch_servers_midway.sh` — submits one `vllm serve` job per model via
  `sbatch --wrap`, encodes the port in the job name as `model_key:port`, then
  submits the experiment job as a dependent (`--dependency=after:<jids>`).
- `slurm/run_exp_midway.sbatch` — the dependent job: polls
  `python -m slurm.ping_servers` until servers answer, runs
  `python run.py --config-name=local_midway +get_models_from_slurm=true`, then
  `scancel`s the server jobs to free the GPUs.
- `slurm/probe_vllm_013.sbatch` — a tiny single-GPU job that proves the toolchain
  (torch sees the GPU, vLLM loads a 0.5B model and generates). Run the PBS analog
  of this **first**, before anything else.
- `src/utils/server.py` — `_discover_servers_from_squeue()` parses
  `squeue --me -o '"%j, %N, %T, %i"'` into `{model_key, model_id, urls, job_ids}`,
  building `http://<node>:<port>/v1`. **This is the most scheduler-specific code
  and the trickiest part of the port — see the discovery section below.**
- `slurm/ping_servers.py` and `src/runner.py` (~line 1269, the
  `if cfg.get_models_from_slurm:` block) — these are **scheduler-agnostic**; they
  only call `get_available_servers()`. You should not need to change them.

- **`../MARSHAL/polaris_pbs_notes.md`** — NEW (v2). The proven Polaris cluster facts +
  the gotcha log from a sibling project on this exact allocation. Treat its
  "Cluster facts (Polaris / ALCF)" table and lessons as ground truth.

## ✅ CONFIRMED Polaris facts + proven gotchas (v2 addendum)

> These were verified running real jobs on this allocation (June 2026). They
> **supersede the "Polaris facts to CONFIRM" section below** (kept intact for history).
> ⚠️ MARSHAL is a *training* pipeline (Ray + DeepSpeed + weight-sync + Megatron);
> Decrypto is *inference serving only*. So only the **cluster-level** facts/gotchas
> apply to you — see "What does NOT apply" at the end of this section.

### Confirmed cluster facts (no longer "go check")

| Item | Confirmed value |
|---|---|
| **Account (`-A`)** | **`lighthouse-uchicago`** ⚠️ NOT "Uchicago-lighthouse" (PBS rejects that). Verify with `sbank-list-allocations`; thousands of node-hours free. |
| **Queue (`-q`)** | **`debug`** for the smoke (1–2 nodes, ≤1 h). Also `debug-scaling`, `prod`. |
| **Filesystems** | **`-l filesystems=home:eagle`** — REQUIRED; PBS rejects jobs without it. |
| **Select line** | `-l select=1:ncpus=64:ngpus=4` (do NOT add `:system=polaris`). |
| GPUs | **4× NVIDIA A100, 40 GB each** (sm_80) — tight vs Midway's 140 GB H200. |
| CPU/RAM | AMD EPYC Milan, 64 hardware threads, 512 GiB. |
| **Conda module** | `module use /soft/modulefiles && module load conda/2025-09-25 && conda activate base` (build/activate your venv on top). Replaces Midway's miniforge/mamba line. |
| Native CUDA toolkit | `/soft/compilers/cudatoolkit/cuda-12.4.1` (driver supports ≥ CUDA 12.8). |
| **Node-local SSD** | `/local/scratch` exists, fast + writable (RAID0). Use for `TMPDIR`; fall back to `/tmp`. |
| Member storage root | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/` (repo, venv, models, caches, server-discovery files). |
| Job ids | `1234567.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`; bare number via `${PBS_JOBID%%.*}`. |
| Cross-node (HSN) | Compute nodes named like `x3001c0s7b1n0`; HSN IPs in `10.201.x.x`; FQDN suffix `.hsn.cm.polaris.alcf.anl.gov`. Capture the node's HSN address at server startup for discovery. |
| Containers | Official container path unusable (no `.sif`, registry unreachable). Run **native** in a conda venv (your plan anyway). |

### Gotchas proven on this allocation — these WILL bite you

1. **A venv on `/lus/eagle` (Lustre) makes `import torch`/vllm "HANG" on cold compute
   nodes.** Biggest time-sink in the sibling port. Lustre is fast for big sequential
   reads but terrible for the ~70k-tiny-files access pattern of importing an ML env:
   ~19 min on a cold node (vs 47 s warm) — looks like a hang. **Fix:** build the venv
   once, **tar it**, and in every job **extract the tar to `/local/scratch`** and run
   from there (one big-file read; ~12 s for 8 GB). Activate **manually** (do NOT
   `source bin/activate` — it hardcodes the original path): `export VIRTUAL_ENV=$LOCAL_VENV;
   export PATH=$LOCAL_VENV/bin:$PATH; hash -r`. Working example:
   `../MARSHAL/scripts/train_polaris.pbs`.
2. **Compute nodes have NO outbound internet.** Network calls fail (a stray
   `connect(8.8.8.8)` raised `OSError: Network is unreachable`; HF downloads fail too).
   **Pre-download every model on a login node** into the eagle model store; the smoke
   model is already staged at
   `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/models/Qwen2.5-0.5B-Instruct`.
   In the job: `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, point `HF_HOME` at
   eagle, and give `vllm serve` a **local path**, not a hub id.
3. **TMPDIR hygiene.** `unset TMPDIR SLURM_TMPDIR; export
   TMPDIR=/local/scratch/$USER_${PBS_JOBID%%.*}; mkdir -p`. Point
   `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` at eagle (persistent), not scratch.
4. **`--enforce-eager` for the smoke** (skips torch.compile; avoids autotune-cache
   write problems and speeds first load).
5. **Pin `transformers<5` + `tokenizers<0.22`.** A fresh install pulls transformers 5.x,
   which dropped `all_special_tokens_extended` that vLLM 0.8.x calls → load-time error.
   Sibling env: transformers 4.51.2 / tokenizers 0.21.4 / numpy 1.26.4 / torch 2.6.0+cu124
   / vllm 0.8.4. Don't blindly `pip install -r requirements.txt` (it pins torch 2.9 /
   vllm 0.13 — unvalidated here). Runner side is HTTP-only, so one env hosts both.
6. **A per-job cap of `pids.max = 4096`** (every process+thread in the job). It mauled
   the *training* sibling (Ray fans out dozens of workers). `vllm serve` is far lighter
   (one engine, one worker per GPU) and should fit easily — but if you see
   `pthread_create ... Resource temporarily unavailable` / `OpenBLAS blas_thread_init
   ... failed`, that's this cap; set `OMP_NUM_THREADS=8` (and the OPENBLAS/MKL/NUMEXPR
   family) to cut per-process threads.
7. **PBS prologue / filesystem mounts can be slow or flaky** (20–40 min from `R` to
   script start; during an ALCF incident, jobs can hang the whole walltime with 0-byte
   output). If a job is silent for long, check `pbsnodes -l` for offlined nodes before
   assuming your bug. Stream a live wrap-log
   (`exec > >(stdbuf -oL -eL tee -a $WRAP_LOG) 2>&1`) so you can see where it is.

### What does NOT apply to you (MARSHAL-only, training-specific — ignore)
- DeepSpeed `fused_adam` JIT compile (nvcc + gcc-12) — no training here.
- ROLL/Ray weight-sync into a running engine, and the **vLLM-V1 "can't serialize
  CUDA/bf16 tensor" bug** — that was about *updating* weights live. You load once and
  serve; V1 is fine.
- Distinct-GPU placement / NCCL broadcast, Megatron tensor-parallel internals,
  `RAY_NUM_CPUS`, env-worker fan-out — Ray/training concepts. For TP, just use
  `vllm serve --tensor-parallel-size N`; no Ray.

## The Slurm→PBS porting surface

| Concern | Midway (Slurm) | Polaris (PBS Pro) — what you need to do |
|---|---|---|
| Submit a job | `sbatch script` / `sbatch --wrap "..."` | `qsub script` (PBS has **no `--wrap`** — write the server command into a real `.pbs` script, or `qsub -` with a heredoc) |
| Directives | `#SBATCH ...` | `#PBS ...` |
| Account/allocation | `--account=rcc-staff` | `-A <project>` (your ALCF project/allocation name) |
| Partition/queue | `--partition=test` | `-q <queue>` (use `debug` for the smoke test) |
| Walltime | `--time=02:00:00` | `-l walltime=01:00:00` |
| GPUs + node shape | `--gres=gpu:N --nodes=1 --cpus-per-task=K --mem=...` | `-l select=1:ngpus=N:ncpus=K` |
| **Filesystems** | (implicit) | `-l filesystems=home:eagle` — **Polaris rejects jobs that don't declare this**; include every FS the job touches |
| List my jobs | `squeue --me` | `qstat -u $USER` / `qstat -f <jobid>` |
| Cancel | `scancel <jid>` | `qdel <jid>` |
| Job dependency | `--dependency=after:<jid>` | `qsub -W depend=after:<jid> ...` (PBS supports `after`, `afterok`, `afterany`) |
| Parse a submitted job id | `--parsable` returns bare id | `qsub` already prints the job id to stdout (capture it; it looks like `1234567.polaris-pbs-...`) |
| Pass env into job | `--export=ALL,FOO=bar` | `qsub -v FOO=bar` (and `-V` to export the full env) |
| Output/error files | `--output=...%j.out` | `#PBS -o ... -e ...` (PBS uses `$PBS_JOBID`, not `%j`) |
| Job-local id in script | `$SLURM_JOB_ID` | `$PBS_JOBID` |
| Node-local scratch | `/tmp/$USER_$SLURM_JOB_ID` | Polaris nodes have **node-local SSD** (commonly `/local/scratch`) — use it for `TMPDIR`; confirm the path on a compute node |

## The server-discovery gotcha (read carefully)

The Midway path encodes the port in the Slurm **job name** (`model_key:port`) and
recovers the server's node from `squeue`'s `%N` column. **This does not port
cleanly to PBS**: PBS job names have length limits and disallow characters like
`:`, so you cannot reliably stuff `model_key:port` into `-N`.

`src/utils/server.py` already has a clean, scheduler-agnostic escape hatch — use
it as your **primary** mechanism instead of fighting `qstat`:

```python
# get_available_servers() checks this first:
servers_file = os.environ.get("DECRYPTO_SERVERS_FILE")
if servers_file and os.path.exists(servers_file):
    return _load_servers_from_file(servers_file)   # reads JSON: [{model_key, model_id, urls, job_ids}, ...]
```

Recommended PBS design:

1. The **server job**, once `vllm serve` reports `Application startup complete`,
   appends its own `{model_key, model_id, urls: ["http://<this-node>:<port>/v1"], job_ids: ["<PBS_JOBID>"]}`
   to a JSON file on shared storage (e.g. `$REPO_ROOT/logs/servers/<jobid>.json`,
   or a single merged file). Get the routable node address with `hostname -f` /
   `$(hostname).hsn.cm.polaris.alcf.anl.gov` — confirm which hostname compute
   nodes use to reach each other over the high-speed network.
2. The **experiment job** sets `DECRYPTO_SERVERS_FILE` to that file and runs
   `run.py +get_models_from_slurm=true` (the flag name still works — it just
   means "discover servers" — but you may add a clearer alias if you like).
3. This sidesteps `qstat` parsing entirely and is robust to PBS job-name rules.

If you'd rather mirror Midway exactly, add `_discover_servers_from_qstat()` that
parses `qstat -f` for job name + `exec_host`, but the file-based approach is
simpler and less brittle. Either way, **keep `get_available_servers()` working
for both clusters** — don't break the Slurm path.

Note `ping_servers.py` and `runner.py` call `get_available_servers()` unchanged,
so once discovery works they need no edits. (`python -m slurm.ping_servers` still
works regardless of where you put the PBS scripts, since it's imported by module
path.)

## Polaris facts to CONFIRM on the cluster (don't trust these blindly)

I'm authoring this from Midway and cannot see Polaris. Verify each of these
yourself (e.g. `qstat -Q`, `nvidia-smi`, ALCF docs) and record them in your notes:

- **Allocation/project name** for `-A` (you need an active Polaris allocation).
- **Queue** — use `debug` for the smoke test (fast turnaround, ≤1 hr, 1–2 nodes).
  Confirm limits with `qstat -Q`.
- **`-l filesystems=` value** — almost certainly `home:eagle` (or `home:grand`).
  Jobs are rejected without it.
- **GPUs** — Polaris nodes are **4× NVIDIA A100 (40 GB each per ALCF docs)**.
  Confirm with `nvidia-smi`. **This is tighter than Midway's H200 (~140 GB
  each):** a 70B model at bf16 (~140 GB) needs TP=4 across the whole node and
  may need `--gpu-memory-utilization` tuning. **Start the smoke test with a
  small model (e.g. Qwen2.5-0.5B) at TP=1** to prove the pipeline, then scale.
- **Conda/module setup** — Polaris uses something like
  `module use /soft/modulefiles && module load conda && conda activate <env>`.
  Confirm the exact incantation; it replaces Midway's
  `module load python/miniforge-25.3.0 && mamba activate ...`.
- **Project storage root** — e.g. `/eagle/<Project>/<user>/...` or `/grand/...`.
  Repoint the Midway `/project/rcc/mehta5/...` paths (repo root, logs, HF cache,
  torch-inductor cache, model cache) to your Polaris equivalents.
- **Cross-node addressing** — confirm the hostname/interface compute nodes use to
  reach each other (HSN), since the experiment job connects to the server job's
  node over HTTP.

## Carry-over gotchas from the Midway port (likely apply here too)

- **`vllm serve` needs `--enforce-eager` for the smoke test.** On Midway,
  `torch.compile`'s autotune cache tried to write into a stale scratch path and
  killed the engine on load. `--enforce-eager` skips compile; revisit for perf.
- **Scrub inherited TMPDIR.** Inside the job: `unset TMPDIR`, then set
  `TMPDIR=<node-local-scratch>/$USER_$PBS_JOBID` and `mkdir -p` it. Point
  `TORCHINDUCTOR_CACHE_DIR` at persistent project storage.
- **Do NOT `pip install -r requirements.txt`.** It pins `torch==2.9.0` /
  `vllm==0.13.0`. Use whatever vLLM/torch build is validated for Polaris's
  driver/CUDA. The runner side is **HTTP-only and never imports torch/vllm**, so
  a single env can host both server and runner (install runner deps:
  `dotenv nltk gensim scipy pandas tqdm anthropic openai requests hydra-core litellm`).
- On Midway, `transformers<5` and `tokenizers<0.22` were needed to avoid an
  `all_special_tokens_extended` AttributeError. Watch for the same.
- Add any new served model's `model_key → local path` to `agent_paths` in
  `src/utils/server.py`, and keep the config's `model_key`/`model_id`/
  `fixed_interceptor` consistent with it.

## Deliverables

Mirror the Midway naming so the two ports sit side by side:

1. `slurm/probe_vllm_polaris.pbs` — single-GPU toolchain probe (PBS analog of
   `probe_vllm_013.sbatch`). **Run and pass this first.**
2. `slurm/launch_servers_polaris.sh` — `qsub` launcher: one server job per model,
   then the dependent experiment job. (Or put PBS files in a new `pbs/` dir —
   your call; `slurm/` is just a name and module imports won't break.)
3. `slurm/run_exp_polaris.pbs` — dependent experiment job: wait for servers
   (poll `python -m slurm.ping_servers`), run `run.py`, then `qdel` the servers.
4. `config/examples/local_polaris.yaml` — config analogous to `local_midway.yaml`
   (self-play, one episode, `match_encoder_decoder: true`, one `LocalModel`).
   **Do not reuse `argonne_polaris.yaml` — that's the gateway path.**
5. `src/utils/server.py` — add PBS-compatible discovery (prefer the
   `DECRYPTO_SERVERS_FILE` approach) **without breaking the Slurm path**.
6. **`polaris_pbs_notes.md`** — the PBS equivalent of `midway_notes.md`.
   ⚠️ **Use this filename, NOT `midway_alcf_inference_notes.md`** — that file
   already exists and documents the unrelated inference-gateway path.

## Validation ladder

1. **Probe** — submit `probe_vllm_polaris.pbs`; confirm `nvidia-smi`, torch sees
   the GPU, vLLM loads a 0.5B model and generates. Don't proceed until green.
2. **Smoke test (small)** — launch one small model (TP=1) + dependent experiment;
   confirm: server comes up → `ping_servers` gets a healthy reply → `run.py`
   discovers the server → one self-play episode completes → results written to
   `results/<exp_name>/experiment_summary.csv` → servers `qdel`'d.
3. **Scale (optional)** — repeat with a 70B model at TP=4 across the full node,
   tuning `--gpu-memory-utilization` for the 40 GB A100s.

## Working method

- Get the code: `git fetch && git checkout midway-decrypto` (the Midway port and
  these notes live on that branch; the remote is
  `git@github.com:rmehta1987/decrypto.git`). Consider a `polaris-decrypto` branch
  for your work.
- Keep a **decisions/changes log** at the bottom of `polaris_pbs_notes.md` with
  dated entries and job ids, exactly like `midway_notes.md` does — it's the most
  useful artifact when something breaks.
- Prefer writing PBS-flavored copies (`*_polaris.*`) over editing the Midway
  scripts in place, so both ports remain diffable.
- When a Polaris fact contradicts an assumption in this prompt, trust the cluster
  and note the correction.
