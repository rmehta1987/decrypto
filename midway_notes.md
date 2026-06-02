# Running Decrypto on RCC Midway

This is a living document. It explains how to stand up and run the Decrypto
experiments on the University of Chicago RCC **Midway** cluster, and it records
the changes that were needed to make the inherited slurm scripts (originally
written for the CMU `ycleong` cluster) work here.

If you just want to run a smoke test, jump to
[Running a smoke test](#running-a-smoke-test). If something breaks, the
[Decisions / changes log](#decisions--changes-log) at the bottom captures the
problems we already hit and how we fixed them.

---

## Cluster facts (Midway / RCC)

These are the hardware and environment facts the scripts depend on.

| Item | Value |
|---|---|
| Account | `rcc-staff` |
| Partition available to us | `test` only |
| GPU we target | NVIDIA H200 (constraint `H200`) |
| GPU VRAM observed | ~140 GiB free on the H200 (probe log 50177021) |
| Driver | 535.216.03 (max CUDA 12.2 advertised) |
| Toolchain in the env | torch 2.8.0+cu128, vllm 0.10.2 — works via CUDA Minor-Version Compatibility |
| Python module | `python/miniforge-25.3.0` |
| Conda env | `/project/rcc/mehta5/conda-envs/vllm-probe` |
| Project root | `/project/rcc/mehta5/decrypto` |
| Model cache | `/project/rcc/mehta5/vllm/models/` |

For confirming the cluster works, we use **1 node × 4 H200**. That is enough for
every smoke-test path: a small served model with a baseline opponent, two small
models side by side, or a 70B model at tensor-parallel 4. Questions about
partition caps and multi-node serving are deferred — we don't need them to prove
the pipeline runs.

---

## One-time setup

You only need to do this once per machine/account.

1. **Activate the environment** every time you open a new shell. Use the mamba
   pattern below — do **not** use `source activate`, which falls through to the
   system anaconda 3.8 on the login nodes and will not have the right packages:

   ```bash
   module load python/miniforge-25.3.0
   eval "$(mamba shell hook --shell bash)"
   mamba activate /project/rcc/mehta5/conda-envs/vllm-probe
   ```

2. **Trust the existing package versions.** The `vllm-probe` env already has a
   working stack: torch 2.8.0+cu128, vllm 0.10.2, transformers (pinned `<5`),
   and tokenizers (pinned `<0.22`). Do **not** run
   `pip install -r requirements.txt` — that file pins `torch==2.9.0` and
   `vllm==0.13.0`, which are incompatible with the cluster's NVIDIA 535 driver
   and will break the env.

3. **If you ever need to recreate the runner-side dependencies** (the experiment
   side talks to vLLM over HTTP and does not import torch/vllm), install just
   these into the env:

   ```bash
   pip install dotenv nltk gensim scipy pandas tqdm anthropic openai requests hydra-core litellm
   ```

4. **Download any model you plan to serve** into the model cache. For example:

   ```bash
   huggingface-cli download Qwen/Qwen2.5-72B-Instruct \
     --local-dir /project/rcc/mehta5/vllm/models/Qwen2.5-72B-Instruct
   ```

   The model's `model_key` must already exist in `agent_paths` in
   `src/utils/server.py`, pointing at this local path. The keys validated so far
   are `qwen2.5_0.5B`, `qwen2.5_72B`, and `llama3.1_70B`.

---

## Running a smoke test

The launcher does almost everything for you: it submits the vLLM server job(s)
and then submits the experiment job as a *dependent* job that waits for the
servers, runs the games, writes results, and cancels the servers to free the
GPU. You normally only run a single command.

1. **Pick the model to serve.** Open `slurm/launch_servers_midway.sh` and edit
   the `models` and `ngpus` arrays near the top. Each entry is
   `"model_key:model_path"`, and `ngpus` is the tensor-parallel size (one entry
   per model):

   ```bash
   models=( "llama3.1_70B:/project/rcc/mehta5/vllm/models/Meta-Llama-3.1-70B-Instruct" )
   ngpus=(4)   # 70B at bf16 ~140 GB → spread across the whole 4× H200 node
   ```

2. **Point the experiment config at the same model.** In
   `config/examples/local_midway.yaml`, make sure `fixed_interceptor`,
   `models[].model_key`, and `models[].model_id` all refer to the model you just
   chose. The `model_key` must match the slurm job-name prefix and the
   `agent_paths` entry in `src/utils/server.py`.

3. **Submit the launcher** from the project root:

   ```bash
   cd /project/rcc/mehta5/decrypto
   bash slurm/launch_servers_midway.sh
   ```

   This prints the server job ID(s) and the port each model is served on, then
   submits the dependent experiment job.

4. **Watch the server come up.** Tail its log until you see
   `Application startup complete`:

   ```bash
   tail -f logs/vllm/<model_key>-<jid>.out
   ```

5. **Let the experiment job run.** Once the server is up, the experiment job
   polls `python -m slurm.ping_servers` until it gets a healthy reply, then runs
   `run.py` with `config-name=local_midway` and `get_models_from_slurm=true` (so
   it discovers the server from the slurm queue). Watch its log here:

   ```bash
   tail -f logs/paper/midway_smoke_<jid>.out
   ```

6. **Collect the results.** When the run finishes it writes
   `results/midway_smoke/experiment_summary.csv` and automatically `scancel`s the
   server job to release the GPU. A smoke test is one episode and finishes in
   roughly a minute once the model is loaded.

---

## What changed from the inherited scripts

The original scripts targeted the CMU `ycleong` cluster. Rather than edit them in
place, we wrote Midway-flavored copies (`*_midway.*`) and kept the originals for
easy diffing. The substantive changes were:

- **Scheduler settings.** `--partition=general` → `--partition=test`, added
  `--account=rcc-staff`, and `--constraint="a100|h100"` → `--constraint=H200`.
  We also lowered the wall time to `02:00:00` to fit the `test` partition.
- **Paths.** All CMU NFS paths (`/net/projects2/ycleong/...`) were repointed to
  Midway project storage under `/project/rcc/mehta5/...`, including the repo
  root, log directories, the Hugging Face cache, and the torch-inductor cache.
- **Environment activation.** The CMU conda activation was replaced with the
  mamba pattern, and the `vllm serve` command is wrapped so it loads the module
  and activates the env *inside* the slurm job.
- **Models.** The original model list referenced a 70B model plus local
  checkpoints that don't exist here. We started with a single small validated
  model and grew from there.
- **`src/utils/server.py`.** Added `agent_paths` entries (`qwen2.5_0.5B`,
  `qwen2.5_72B`, local `llama3.1_70B`) so the squeue-based server discovery can
  resolve a job name to a local model path.

---

## Results so far

All three models and both tensor-parallel configs have been confirmed
end-to-end on Midway:

- **Qwen2.5-0.5B** (1× H200) — pipeline runs, but the model is too small to play
  well (it fails the JSON-format retries, so gameplay is garbage). This is a
  model-capacity issue, not a pipeline issue, and is expected at this scale.
- **Qwen2.5-72B** (TP=2, 2× H200) — plays competent Decrypto with zero
  JSON-format failures: plausible one-word hints, correct decoding, intelligent
  intercepts.
- **Llama-3.1-70B** (TP=4, whole node) — plays at the same level as Qwen2.5-72B,
  zero JSON-format failures.

The takeaway: the orchestration path (launch → server up → `ping_servers` OK →
`run.py` discovers the server → episode completes → results written → GPU freed)
works, and real runs just need a sufficiently large model.

---

## Decisions / changes log

- **2026-05-27 — Pinned transformers `<5` and tokenizers `<0.22`.** Fixes the
  `all_special_tokens_extended` AttributeError; the probe passes afterward.
- **2026-05-27 — Standardized on mamba activation.** `source activate` falls
  through to the system anaconda 3.8 on login nodes.
- **2026-05-27 — Smoke test v1 (server 50177252, exp 50177253).** The vLLM
  server came up and `ping_servers` got a 200 OK, but the experiment job died at
  `import hydra`: the env only had the vLLM stack, none of Decrypto's runner
  deps. Installed the runner deps individually (see step 3 of setup) rather than
  using `requirements.txt`, which pins incompatible torch/vllm. A single env now
  hosts both server and runner, which is fine because the runner is HTTP-only and
  never imports torch/vllm.
- **2026-05-27 — Smoke test v2 (server 50177369).** vLLM crashed during the
  `torch.compile` autotune cache save with a `PermissionError` writing to a
  `/scratch/local/jobs/...` path. Root cause: `--export=ALL` propagated a stale
  `TMPDIR` from the submitting shell (pointing at a previous, cancelled job's
  scratch). Fixes, applied to both Midway scripts:
  1. `unset TMPDIR SLURM_TMPDIR`, then set `TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}`
     inside the job.
  2. Set `TORCHINDUCTOR_CACHE_DIR` to project storage so the inductor cache
     survives outside transient scratch.
  3. Added `--enforce-eager` to skip `torch.compile` for the smoke test (mirrors
     the working probe). Revisit once we want full performance.
- **2026-05-27 — Smoke test GREEN (server 50177648, exp 50177649).** Qwen2.5-0.5B
  ran one episode end-to-end and wrote `results/midway_smoke/experiment_summary.csv`;
  cleanup `scancel` released the GPU. Gameplay was garbage (0.5B fails the JSON
  retries) — a model-capacity issue, not a pipeline one.
- **2026-05-27 — Retargeted the second smoke test to Qwen2.5-72B.** The
  Llama-3.1-70B access request was pending Meta review, so we used the
  ungated, same-scale Qwen2.5-72B. Pre-downloaded it, added `qwen2.5_72B` to
  `agent_paths`, and set the launcher to TP=2 / `--gres=gpu:2` / `--mem=128G`.
- **2026-05-27 — Qwen2.5-72B smoke test GREEN (server 50180980, exp 50180981).**
  TP=2 across 2× H200, `--enforce-eager`. One episode (4 turns, ~67s): zero
  JSON-format failures, Eve cracked turns 3 and 4 and won. The 72B model plays
  competent Decrypto.
- **2026-05-27 — Llama-3.1-70B smoke test GREEN (server 50185192, exp 50185193).**
  TP=4 across the whole 4× H200 node, `--enforce-eager`. One episode (3 turns,
  ~61s): zero JSON-format failures, Eve cracked turns 2 and 3 and won. Plays at
  the same level as Qwen2.5-72B. With this, all three models and both TP configs
  are proven on Midway.
