# Midway port — running notes

Living doc. Tracks hardware facts about RCC Midway and the analysis/changes
needed to make the inherited Decrypto slurm scripts (written for the CMU
`ycleong` cluster) run here.

---

## Cluster facts (Midway / RCC)

| Item | Value |
|---|---|
| Account | `rcc-staff` |
| Partition available to us | `test` only |
| GPU we are targeting | NVIDIA H200 (constraint `H200`) |
| GPU VRAM observed | ~140 GiB free on the H200 (probe log 50177021) |
| Driver | 535.216.03 (max CUDA 12.2 advertised) |
| Toolchain bundled in env | torch 2.8.0+cu128, vllm 0.10.2 — works via CUDA Minor-Version Compatibility |
| Python module | `python/miniforge-25.3.0` |
| Activation pattern | `eval "$(mamba shell hook --shell bash)" && mamba activate <env>` (NOT `source activate`) |
| Project root | `/project/rcc/mehta5/decrypto` |
| vllm-probe conda env | `/project/rcc/mehta5/conda-envs/vllm-probe` |
| Model cache | `/project/rcc/mehta5/vllm/models/` (Qwen2.5-0.5B-Instruct already present) |

Resource decision for confirming Midway works:
- **1 node × 4 H200** is the target allocation. Plenty for every smoke-test
  path (small served model + baseline; two small models side-by-side; or a
  70B at TP=4 if we later want to stretch it). Open questions about partition
  caps and multi-node serving are deferred — not needed to prove the cluster
  works.

---

## Env state (vllm-probe) — verified 2026-05-27

```
python       3.12.13
torch        2.8.0+cu128
vllm         0.10.2
transformers 4.x  (pinned <5 — 5.x removes `all_special_tokens_extended`)
tokenizers   <0.22
+ runner deps: dotenv, nltk, gensim, scipy, pandas, tqdm, anthropic,
               openai, requests, hydra-core, litellm
```

Probe `slurm/probe_vllm_013.sbatch` passes Stages 0–3 (nvidia-smi, torch matmul,
vLLM load + generate on Qwen2.5-0.5B). Log: `logs/vllm013-probe-50177021.log`.

Note: do NOT run `pip install -r requirements.txt` against this env —
`requirements.txt` pins `torch==2.9.0` / `vllm==0.13.0` which are incompatible
with the cluster's NVIDIA 535 driver. We satisfy the runner-side deps
individually.

---

## Inherited script audit — what needs changing

The repo's slurm scripts target the CMU `ycleong` cluster and need surgical
edits before they will run here.

### `slurm/launch_servers.sh`
- [ ] `--partition=general` → `--partition=test`
- [ ] Add `--account=rcc-staff`
- [ ] `--constraint="a100|h100"` → `--constraint=H200`
- [ ] Logs path: `/net/projects2/ycleong/sg/strategy-rl/decrypto/logs/vllm/...`
      → `/project/rcc/mehta5/decrypto/logs/vllm/...` (and `mkdir -p`)
- [ ] `HF_HOME` / `HUGGINGFACE_HUB_CACHE` paths point at the CMU NFS — repoint
      to a project-local dir (e.g. `/project/rcc/mehta5/hf_cache`)
- [ ] `--wrap "vllm serve ..."` runs in whatever env the submitting shell has;
      needs to first `module load python/miniforge-25.3.0 && mamba activate
      /project/rcc/mehta5/conda-envs/vllm-probe` inside the wrap
- [ ] Models list currently has 70B + Qwen3-4B + a `/net/projects2/...` local
      checkpoint that doesn't exist here. Start with a single small model
      (Qwen2.5-0.5B already validated) for the first end-to-end.
- [ ] `--time=12:00:00` likely exceeds `test`-partition cap — confirm and lower

### `slurm/run_exp.sbatch`
- [ ] Same partition / account / time / log-path fixes
- [ ] `cd /net/projects2/ycleong/sg/strategy-rl/decrypto` → `cd /project/rcc/mehta5/decrypto`
- [ ] Conda activation: replace
      `source /net/projects2/ycleong/sg/miniconda3/etc/profile.d/conda.sh && conda activate decrypto`
      with the mamba pattern + our env path
- [ ] Config referenced (`figure_4_tom_piaget`) requires `qwen3_4b` and a
      `qwen3_4b_hanabi` local checkpoint not present here — use/author a smaller
      config for first run (single small served model + a baseline)

### `slurm/run_all.sbatch` (newer all-in-one)
Same fixes as above plus its `NODE_GPU_CAPACITY=8` assumption needs revisiting
once we know what nodes the `test` partition actually grants.

---

## Plan to first green end-to-end run on Midway

Allocation: 1 node × 4 H200, `--partition=test --account=rcc-staff`.

1. **Smoke test (uses 1 of 4 GPUs).** Serve one `Qwen2.5-0.5B-Instruct`
   (already on disk, validated by probe); experiment side uses a
   `BaselineModel` (GloVe) opponent. Proves the orchestration path:
   `launch → server up → ping_servers OK → run.py reads slurm queue → 1
   episode completes → output file written`.
2. **Two-model test (uses 2 of 4 GPUs).** Serve Qwen3-4B as encoder/decoder
   (match_encoder_decoder) + Qwen3-4B as fixed interceptor. Mirrors the
   shape of `figure_4_tom_piaget_test` without needing the missing
   `qwen3_4b_hanabi` checkpoint.
3. After (1) and (2) pass on the same allocation → Midway is confirmed; defer
   the partition wall-time / multi-node questions until we actually need
   bigger runs.

Concretely, work items:
- Author `slurm/launch_servers_midway.sh` and `slurm/run_exp_midway.sbatch`
  (Midway-flavored copies; keep originals for diff visibility).
- Author `config/examples/local_midway.yaml` (smoke test) and a small
  `config/paper/figure_4_tom_piaget_midway.yaml` (two-model test).
- Make sure `logs/vllm/` and `logs/paper/` directories exist before submit.

---

## Decisions / changes log

- 2026-05-27 — Capped transformers to `<5` to fix `all_special_tokens_extended`
  AttributeError; tokenizers capped to `<0.22` to match. Probe now passes.
- 2026-05-27 — Standardized on mamba activation; `source activate` falls
  through to system anaconda 3.8 on login nodes.
- 2026-05-27 — Smoke test v1 (server jid 50177252, exp jid 50177253): vLLM
  server came up cleanly and `ping_servers` got a 200 OK; the experiment job
  died at `import hydra`. The `vllm-probe` env only had the vLLM stack — none
  of Decrypto's runner deps. Cannot just `pip install -r requirements.txt`
  because that file pins `torch==2.9.0` and `vllm==0.13.0` (incompatible with
  driver 535 / what works here). Installed only the runner-side deps into
  `vllm-probe`:
  ```
  pip install dotenv nltk gensim scipy pandas tqdm anthropic openai requests hydra-core litellm
  ```
  Single env (`vllm-probe`) now hosts both the server and the runner, which is
  fine because the runner is HTTP-only and doesn't touch torch/vllm at import
  time.
- 2026-05-27 — Added `qwen2.5_0.5B` entry to `src/utils/server.py:agent_paths`
  so squeue-based discovery resolves the job name to the local model path.
- 2026-05-27 — Smoke test v2 (server jid 50177369): vLLM crashed during
  `torch.compile` autotune cache save with `PermissionError: ... /scratch/local/jobs/50177252`.
  Root cause: `--export=ALL` propagated a stale `TMPDIR` from the submitting
  shell (pointing at the prior cancelled job's SLURM scratch) into the new
  job. Fixes applied to `launch_servers_midway.sh` (and mirrored in
  `run_exp_midway.sbatch`):
  1. `unset TMPDIR SLURM_TMPDIR` then re-set `TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}`
     inside the wrap.
  2. Set `TORCHINDUCTOR_CACHE_DIR=/project/rcc/mehta5/torchinductor_cache`
     so the inductor cache lives in project storage, not transient SLURM scratch.
  3. Added `--enforce-eager` to the vLLM serve flags — skips torch.compile
     altogether for the smoke test (mirrors what the working probe used).
     Will revisit once the smoke test is green and we want full perf.
- **2026-05-27 — SMOKE TEST GREEN (server jid 50177648, exp jid 50177649).**
  Server reached "Application startup complete" on `midway3-0606`, `ping_servers`
  got a 200 OK, `run.py` ran one episode of Qwen2.5-0.5B vs itself, produced
  `results/midway_smoke/experiment_summary.csv`, and the cleanup `scancel`
  released the GPU. Total exp wallclock: 6 sec. End-to-end Decrypto on Midway
  is confirmed working.

  (Sidebar: the 0.5B model failed every JSON-format retry, so gameplay was
  garbage — that's a model-capacity issue, not a pipeline issue, and is
  expected at this scale. Real runs need a bigger model.)

- 2026-05-27 — Smoke v2 retarget: Llama-3.1-70B access request is pending
  Meta review, so switching to **Qwen2.5-72B-Instruct** (not gated, same
  scale). Pre-download via
  `huggingface-cli download Qwen/Qwen2.5-72B-Instruct --local-dir /project/rcc/mehta5/vllm/models/Qwen2.5-72B-Instruct`.
  Scripts updated: `agent_paths` adds `qwen2.5_72B → local path`; launcher
  now requests TP=2 / `--gres=gpu:2` / `--mem=128G` / `--time=02:00:00`;
  config uses `max_tokens: 1000`. Plan once Meta approves Llama: load it on
  the full node (TP=4 across all 4 H200s on one node) — will wait longer
  for the resources but only need a single H200-node allocation.

- **2026-05-27 — QWEN2.5-72B SMOKE TEST GREEN (server jid 50180980, exp jid 50180981).**
  TP=2 across 2× H200, `--enforce-eager`. Server loaded weights and reached
  "Application startup complete"; experiment ran 1 episode (4 turns, ~67s).
  Results: 0 miscommunications, 1 intercept (Eve cracked turn 3 and 4),
  Eve wins. Zero JSON-format failures — massive improvement over the 0.5B
  run. The 72B model plays competent Decrypto: generates plausible one-word
  hints, decodes correctly, and intercepts intelligently.
  Results: `results/midway_smoke/experiment_summary.csv`.

- **2026-05-27 — LLAMA-3.1-70B SMOKE TEST GREEN (server jid 50185192, exp jid 50185193).**
  TP=4 across 4× H200 (whole node), `--enforce-eager`. Experiment ran 1
  episode (3 turns, ~61s). Results: 0 miscommunications, 1 intercept
  (Eve cracked turns 2 and 3), Eve wins. Zero JSON-format failures.
  Llama plays competently at the same level as Qwen2.5-72B.
  All three models (0.5B, 72B, 70B) and both TP configs (TP=2, TP=4)
  now proven on Midway end-to-end.
