# Handoff: scale Decrypto's Polaris vLLM pipeline to 3-model cross-play (Llama-3.1-70B + Qwen3-8B + Qwen3-4B)

> Paste everything below the line into a fresh Claude Code session running on an
> ALCF **Polaris** login node, inside this repo checkout (branch
> `polaris-decrypto`). It is written **to that Claude**. The single-model
> self-hosted vLLM pipeline is already proven GREEN on Polaris (job 7186966); your
> job is to scale it to serve and **cross-play three larger models** and to record
> the work to a standard a scientific reviewer would accept.

---

You are a Claude Code session on **ALCF Polaris** (PBS Pro: `qsub` / `qstat` /
`qdel`; login host `polaris-login-04`). The Decrypto self-hosted vLLM serving
pipeline already runs end-to-end here with a 0.5B model on one A100 node. Your
mission is to take it to a real experiment.

## Mission

1. **Serve three models concurrently** on Polaris compute nodes via `vllm serve`:
   - `llama3.1_70B` — Meta-Llama-3.1-70B-Instruct
   - `qwen3_8b` — Qwen3-8B
   - `qwen3_4b` — Qwen3-4B
2. **Run a full cross-play matrix.** All three models are available to all three
   Decrypto roles, so the runner's `itertools.product(encoder, decoder,
   interceptor)` yields **3 × 3 × 3 = 27 role combinations per `env_seed`**. No
   `fixed_*` flags, `match_encoder_decoder: false`. This is the "figure out the
   game" experiment.
3. **Figure out tensor parallelism.** A single A100 holds 40 GB; a 70B model in
   bf16 does not fit one card. You must work out — and *verify on the cluster* —
   the tensor-parallel (TP) size for each model and how the three servers pack
   onto the larger-partition nodes you now have access to. This is the load-bearing
   technical problem; see Objective 2.
4. **Escalate queues in two phases.** Prove each piece with **quick smoke tests on
   `debug`** (fast turnaround, single job, ≤1 h), and only once everything is shown
   working, run the **full 8–16 h pipeline** on a long-walltime queue that allows the
   multi-job server+experiment pattern. `debug` caps a user at 1 running + 1 queued
   job, so it hosts the single-job probes/smokes but not the multi-server run. See
   Objective 3 for the queue map.
5. **Document to reviewer standard.** Keep `polaris_pbs_notes.md` (the lab
   notebook) and `polaris_setup_guide.md` (the recipe) current as you go — see
   **Documentation discipline**, which is a graded deliverable, not an afterthought.

## Do not conflate the two "Polaris" paths in this repo

There are two unrelated ways this repo can use ALCF; both say "Polaris":

- **Self-hosted vLLM serving via PBS** (your task): you `qsub` jobs that run
  `vllm serve` on Polaris A100s, and a Decrypto experiment job drives them over
  HTTP. Direct descendant of the Midway Slurm port.
- **ALCF inference *gateway*** (NOT your task): `config/examples/argonne_polaris.yaml`,
  `src/utils/inference_auth_token.py`, `src/utils/globus_login.py`,
  `src/utils/list_argonne_endpoints.py`, the `use_globus_auth` field. HTTPS calls
  to models ALCF already hosts; blocked on Globus. **Do not touch or rely on it.**

If you find yourself editing anything Globus / `inference_auth_token`-related, stop
— you are in the wrong path.

## Read these first (the proven artifacts)

The single-model port is your template; reuse the working **scripts and config**,
do not rewrite them. The two **notes files** in this table (`polaris_pbs_notes.md`,
`polaris_setup_guide.md`) are the deliberate exception — you will *revise* them per
**Documentation discipline**, not preserve them verbatim.

| File | What it is | Use it for |
|---|---|---|
| `polaris_pbs_notes.md` | Lab notebook: cluster facts, results, decisions log | Ground truth + the file you extend |
| `polaris_setup_guide.md` | Step-by-step recipe incl. "Scaling to a bigger LLM" | The procedure you scale up |
| `slurm/build_decrypto_serve_venv.sh` | Builds the proven cu124 venv + tarball | Env is reusable as-is for the runner; verify it serves the new models |
| `slurm/probe_vllm_polaris.pbs` | Single-GPU toolchain probe | Adapt to probe each new model's load |
| `slurm/smoke_polaris.pbs` | Single-job (server+experiment) smoke — the `debug` path | Reference; not the multi-model path |
| `slurm/server_vllm_polaris.pbs` | Per-model `vllm serve` job; self-registers via `DECRYPTO_SERVERS_FILE` | The server job you parameterize per model/TP |
| `slurm/launch_servers_polaris.sh` | Two-job launcher: N server jobs + dependent experiment job | The multi-model launcher you extend to 3 models |
| `slurm/run_exp_polaris.pbs` | Dependent experiment job: poll `ping_servers`, run `run.py`, `qdel` servers | The experiment job you point at the new config |
| `config/examples/local_polaris.yaml` | 0.5B self-play config | Template for the new 3-model config |
| `src/utils/server.py` | Discovery: `DECRYPTO_SERVERS_FILE` (file/dir) + `agent_paths` | Holds **stale Midway paths** to fix (Objective 5) |
| `src/runner.py` (~L1269 `get_models_from_slurm`; ~L1318–1420 policy/combination build) | How discovered servers merge into `cfg.models` and how role combinations are generated | Confirm 27-combo behavior; do not change semantics |

The proven sibling cluster-facts record is `../MARSHAL/polaris_pbs_notes.md` —
**confirm that relative path still resolves** before citing it (it is a candidate
stale reference; see Objective 5).

## Confirmed cluster facts (re-verify the ones marked ⟳)

| Item | Value | Note |
|---|---|---|
| Scheduler | PBS Pro — `qsub` / `qstat` / `qdel` | login `polaris-login-04` |
| Account (`-A`) | `lighthouse-uchicago` | NOT "Uchicago-lighthouse" — PBS rejects that |
| Filesystems | `-l filesystems=home:eagle` | REQUIRED on every job |
| Select line | `-l select=N:ncpus=64:ngpus=4` | do NOT add `:system=polaris` |
| GPU | 4× NVIDIA A100 **40 GB** (sm_80) per node | ⟳ **re-confirm on the larger-partition nodes with `nvidia-smi`, cross-checked against the [compute-nodes doc](https://docs.alcf.anl.gov/polaris/#polaris-compute-nodes) — if they are 80 GB A100s, the whole TP plan changes** |
| CPU / RAM | AMD EPYC Milan, 64 threads, 512 GiB | |
| Conda | `module use /soft/modulefiles && module load conda/2025-09-25 && conda activate base` | build/activate the relocated venv on top |
| Member base `$BASE` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5` | repo, venv, models, caches |
| Model store | `$BASE/models/` | only `Qwen2.5-0.5B-Instruct` staged today |
| Serving venv + tarball | `$BASE/conda-envs/decrypto-serve` + `$BASE/decrypto-serve-venv.tar` | one env hosts server + HTTP runner |
| Queue (smoke) | `debug` — 1 running + 1 queued/user, ≤1 h, 1–2 nodes | cannot host the two-job pattern |
| Larger queues | ⟳ confirm limits with `qstat -Qf <queue>` | see Objective 3 |
| HSN addressing | nodes `x3xxxc0sxbxnx`; HSN IPs `10.201.x.x`; FQDN `<node>.hsn.cm.polaris.alcf.anl.gov` | server registers its `10.201.x` IP |
| Bare job id | `${PBS_JOBID%%.*}` | full id `1234567.polaris-pbs-01.hsn…` |

**ALCF Polaris documentation — consult it when you need authoritative cluster
detail** (node hardware, queue policies, modules, filesystems). Prefer these over
guessing, and use them to *confirm* the ⟳ rows above rather than trusting this
prompt:

- Polaris overview / **compute-node hardware** (GPUs, memory, NICs): <https://docs.alcf.anl.gov/polaris/#polaris-compute-nodes>
- **Running jobs** (PBS directives, the queue table + per-queue limits, node counts, walltimes): <https://docs.alcf.anl.gov/polaris/running-jobs/>
- Polaris docs landing page (everything else — modules, data transfer, software): <https://docs.alcf.anl.gov/polaris/>

You have web access; fetch these when a fact here is marked ⟳ or when you hit
something this prompt does not cover. When the docs and this prompt disagree, the
docs win — record the correction in the notes.

Carry-over gotchas that still apply (all already coded into the `*_polaris` scripts
— preserve them): venv must be staged from the tarball to `/local/scratch` and
activated manually (Lustre import storm otherwise); compute nodes are offline
(`HF_HUB_OFFLINE=1`, serve a local **path** not a hub id); scrub `TMPDIR` onto
`/local/scratch`; **unset the ALCF `http_proxy` family + set `no_proxy`** (the proxy
hangs in-cluster HTTP — this was the 900 s health-check failure on job 7186962);
`--enforce-eager` for first loads; pin `transformers<5` / `tokenizers<0.22`; watch
`pids.max=4096`.

---

## Objective 1 — Stage the three models (login node, one-time)

Compute nodes have no internet. Download on a **login node** into `$BASE/models/`,
then serve the local path.

- `Qwen/Qwen3-8B` and `Qwen/Qwen3-4B` are open. `meta-llama/Meta-Llama-3.1-70B-Instruct`
  is **gated** — you need an accepted license and an `HF_TOKEN`; if the download
  401s, that is the cause, not a bug. Record the exact repo id you used.
- Disk: Llama-3.1-70B bf16 is **~141 GB**; Qwen3-8B ~16 GB; Qwen3-4B ~8 GB. Check
  the eagle quota *before* pulling 70B (`du -sh $BASE/models` and your project
  quota tool) — do **not** discover you are out of space mid-download.
- Use `huggingface-cli download <repo> --local-dir $BASE/models/<name>`. After each
  download, list the dir and confirm `config.json` + all `*.safetensors` shards and
  the `model.safetensors.index.json` are present (a truncated download is a classic
  silent failure).

## Objective 2 — Figure out tensor parallelism (the core problem)

Do not guess TP sizes — derive them, then **verify each on the cluster** and record
the evidence (job id + the line in the vLLM log that proves it).

**The two hard constraints**

- **Memory.** Per-GPU weight footprint ≈ `2 × params_bytes / TP` (bf16). It must
  leave room on each card for the KV cache + activations + CUDA context (~1–2 GB).
  On 40 GB cards:

  | Model | Params | bf16 weights (approx) | TP | ≈ weights/GPU | Fits 40 GB? | Headroom for KV |
  |---|---|---|---|---|---|---|
  | Qwen3-4B | ~4B | ~8 GB | 1 | ~8 GB | yes | ample (~30 GB) |
  | Qwen3-8B | ~8B | ~16 GB | 1 | ~16 GB | yes | comfortable (~22 GB) |
  | Llama-3.1-70B | ~70B | ~141 GB | **4** | ~35 GB | yes, **tight** | ~3–5 GB/GPU — small KV cache |

  These are starting hypotheses. The 70B case is the risky one: at TP=4 on 40 GB
  cards, weights alone eat most of each GPU. If the engine OOMs on load or reports a
  tiny KV cache, **lower `--max-model-len`** (8192 → 4096 → 2048) and/or nudge
  `--gpu-memory-utilization` toward 0.95. Confirm 70B *actually loads* before you
  build anything on top of it. **TP=4 is the unique single-node fit on 40 GB cards;
  wherever this prompt later writes "TP=4," read it as shorthand for the TP you
  derived and verified here.** If `nvidia-smi` shows 80 GB A100s (the ⟳ caveat in
  the facts table), TP=2 fits 70B and becomes the value to use — re-derive the whole
  table for that case.

- **Head divisibility.** TP must divide **both** `num_attention_heads` and
  `num_key_value_heads` (GQA). Read each model's `config.json` and confirm the TP
  you pick divides both — vLLM will refuse otherwise. Do not hardcode head counts
  from memory; read them from the staged `config.json`.

**Mechanics**

- vLLM flag is `--tensor-parallel-size N`. The N GPUs must be on **one node**.
- **Single-node TP only.** Multi-node TP/PP needs Ray spanning nodes — a real
  bring-up, explicitly out of scope per `polaris_setup_guide.md`. Fit 70B at TP=4
  on one node; do not attempt 2-node serving.
- **Verify, don't assume.** After a server comes up, the vLLM startup log prints the
  available KV-cache size / "GPU blocks". Capture that number per model as the proof
  your TP choice left usable KV space. A server that loads but has ~0 KV blocks will
  fail at request time.

**Co-locating small models on one node (optional optimization).** Qwen3-8B and
Qwen3-4B each use TP=1, so two of them fit on one 4-GPU node. vLLM binds GPU 0 by
default, so a second server on the same node must be pinned with
`CUDA_VISIBLE_DEVICES=1` (and a distinct `--port`) or they collide on GPU 0. The
current `server_vllm_polaris.pbs` runs **one** model per job; the simplest correct
first cut is **one model per node-job** (3 server jobs). Treat co-location as a
follow-up only after the 3-job version is green, and document whichever you choose.

## Objective 3 — Node + queue plan (debug smokes → full 8–16 h run)

Run this in **two queue phases**: cheap verification on `debug`, then the long
production run on a queue built for it. `debug` caps a user at 1 running + 1 queued
job, so it can host the single-job probes/smokes but **not** the multi-server run
(which needs ≥4 concurrent jobs: 3 servers + 1 experiment).

**Phase 1 — quick smokes on `debug`** (≤1 h, single job, fast turnaround):

- Per-model probes (ladder rung 1) and a fused single-job 70B sanity smoke (server +
  a generation in one job, `smoke_polaris.pbs`-style) run here. Use debug to prove
  each model loads, serves, and generates before spending a large allocation.
- What debug *cannot* do: the separate-server + dependent-experiment launch (ladder
  rungs 3–5). That is Phase 2.

**Phase 2 — the full pipeline on a long multi-job queue** (8–16 h):

- **`preemptable` is the fit:** 1–10 nodes/job, ≤72 h walltime, ~20 running+queued
  per project (so the 4-job pattern fits). It **is preemptible**, so add `#PBS -r y`
  (rerun on preemption) and write results incrementally — a kill mid-run must not
  lose the whole matrix.
- **Not suitable for the long run:** `demand` (≤1 h walltime — fine for a short
  concurrent test, not an 8–16 h run, and by-request-only); `prod` (forces ≥10
  nodes/job; we use 1 node/job); `capacity` (168 h and no preemption, but **1 running
  job/user** — single-job only, so it can host the long run *only* if you fuse all
  servers + experiment into one multi-node job, a heavier script change — keep as a
  no-preemption fallback).
- **Confirm the real limits yourself** — both `qstat -Qf <queue>` on the cluster
  *and* the ALCF [running-jobs queue table](https://docs.alcf.anl.gov/polaris/running-jobs/)
  (node min/max, walltime, per-user vs per-project concurrency) — and record them;
  walltimes and limits change.

**Node budget (one-model-per-node-job baseline):** 70B server = 1 node (TP=4);
Qwen3-8B = 1 node; Qwen3-4B = 1 node; experiment = 1 node (CPU/HTTP-only, GPUs idle
but PBS hands out whole nodes). That is **4 jobs / 4 nodes** running at once;
co-locating 8B+4B (Objective 2) drops it to 3 jobs / 3 nodes.

**Wiring the launcher + scripts:**

- Extend `slurm/launch_servers_polaris.sh`: set the `models` / `ngpus` arrays to the
  three models with their TP sizes, switch `#PBS -q` in the server and experiment
  scripts to `preemptable` (and add `#PBS -r y`) for the full run, and verify
  `EXPECTED_MODELS` (derived from the array length, so the experiment waits for all 3
  servers).
- **Walltime:** set the experiment job to your target **8–16 h** and the **server
  walltime ≥ the experiment walltime** so a server never dies mid-game; keep both
  well under the queue's 72 h cap.
- **Raise the timeouts for 70B.** A 70B cold load at TP=4 is minutes, not the 0.5B's
  21 s. Check `HEALTH_TIMEOUT` (server) and `WAIT_TIMEOUT` (experiment) cover it.

## Objective 4 — The cross-play config

Create a new config (e.g. `config/examples/local_polaris_3model.yaml`; do not
overwrite the 0.5B `local_polaris.yaml`). Requirements:

- List all three models under `models:` as `LocalModel` entries. For each, `model_key`
  must match what the server job writes into its discovery JSON, and `model_id` must
  equal the **local path** vLLM serves under.
- `match_encoder_decoder: false`, no `fixed_encoder` / `fixed_decoder` /
  `fixed_interceptor`, empty `filter_model` → full 27-combo matrix per seed.
- **Mind the blow-up:** 27 combos × |`env_seed`| × `num_episodes`. Start with one
  seed and `num_episodes: 1` (27 games) to prove the matrix runs end-to-end, then
  scale seeds/episodes once green. Note the count you ran.
- Point `run_exp_polaris.pbs` (and any smoke) at the new `--config-name` and a fresh
  `exp_name` (e.g. `polaris_3model`) so results land in `results/polaris_3model/`.
- **Qwen3 caveat to watch and record:** Qwen3 are reasoning models that can emit
  `<think>` blocks. If `max_tokens` is too small the model spends its budget thinking
  and never produces the JSON the runner parses → format-retry failures that look
  like a pipeline bug but are a token-budget/thinking-mode issue. If you see this,
  raise `max_tokens` and/or disable thinking, and document it as a model behavior,
  not a pipeline fault (mirrors how the 0.5B's JSON failures were classified).

## Objective 5 — Fix stale references (no broken pointers may remain)

The repo carries references from the RCC Midway origin that are stale on Polaris.
Find and fix them; a reviewer will check that every path and link resolves.

Known stale references to fix:

- `src/utils/server.py` → `agent_paths`: `llama3.1_70B`, `qwen2.5_0.5B`,
  `qwen2.5_72B` point at `/project/rcc/mehta5/vllm/models/...` (**Midway/RCC paths —
  do not exist on Polaris**); `qwen3_4b`, `qwen3_8b`, `llama3.1_8B`, `llama3.2_1B`
  are **bare HF repo ids** (won't resolve on offline compute nodes). Update the keys
  you serve (`llama3.1_70B`, `qwen3_8b`, `qwen3_4b`) to their `$BASE/models/...`
  local paths. **Accuracy note:** `agent_paths` is read only by the legacy
  `_discover_servers_from_squeue()` path, *not* by the file-based
  `DECRYPTO_SERVERS_FILE` path you use — so on Polaris `model_id` comes from the
  servers JSON / config, and `agent_paths` is mostly cosmetic. Fix it anyway (it is
  a stale reference) but do **not** break the Slurm `squeue` branch.
- Doc cross-links: confirm `../MARSHAL/polaris_pbs_notes.md` and every relative link
  inside `polaris_pbs_notes.md` / `polaris_setup_guide.md` actually resolves; fix or
  drop dead ones.
- Scaling text that named `Qwen2.5-72B` / `Llama-3.1-70B` as a *future* target is now
  *realized* — update it to describe what you actually ran.

**Searching the filesystem — important constraint.** Do **not** run `find` over the
cluster root, `$HOME`, or `/lus/eagle/...` broadly — those are large shared Lustre
trees and a wide `find` is punishingly slow and disruptive. Scope every search to
the repo and prefer fast tools: `git grep` / `rg` inside the repo, `find` only
under a specific small subdir (e.g. `find logs -maxdepth 2`), and `git ls-files`
to enumerate tracked files. The same rule holds for staging checks — `ls` a known
path, never sweep the tree.

## Objective 6 — Validation ladder

Climb in order; do not skip a rung. **Commit after each rung passes** (see Working
method) so a later failure reverts cleanly. The `[queue]` tag on each rung follows
the two-phase plan in Objective 3.

1. **[debug] Probe each model in isolation.** Adapt `probe_vllm_polaris.pbs` to load
   each model at its candidate TP and generate one completion. This is where you
   confirm the 70B TP fit and the head-divisibility math — cheaper to fail here than
   in a multi-node launch.
2. **[debug] Single 70B server, fused sanity smoke.** In one job, bring up only the
   70B server, confirm health + a real generation, and capture the KV-cache log line.
   70B is the one most likely to OOM; isolate it before spending a big allocation.
3. **[preemptable] All three servers up.** Launch the 3 server jobs via the extended
   launcher; confirm `ping_servers` reports 3 healthy replies and the discovery dir
   holds 3 `<jobid>.json` files. (Multi-job — cannot run on debug.)
4. **[preemptable, short walltime] Orchestration smoke.** Run the dependent
   experiment over the 27-combo matrix at **1 seed, 1 episode** (27 games)
   end-to-end; confirm `results/polaris_3model/experiment_summary.csv` has all
   combinations and the servers are `qdel`'d at the end. This proves the full
   pipeline cheaply before committing to the long run.
5. **[preemptable, 8–16 h] Full pipeline.** Only once rung 4 is green, scale
   `env_seed` and `num_episodes` to the real workload, raise the walltime to 8–16 h
   (server ≥ experiment), add `#PBS -r y`, and launch the production run. Verify
   results land incrementally so a preemption is recoverable.

---

## Documentation discipline (graded deliverable)

Update **`polaris_pbs_notes.md`** (lab notebook) and **`polaris_setup_guide.md`**
(recipe) as you work. Both currently use ✅/✗/GREEN-banner glyphs; **convert them and
write everything new in the register below.**

**Register — write as if a scientific reviewer and a future reader will grade it:**

- Neutral, precise, evidence-anchored prose. Every claim ties to a **job id and a
  log path**. Prefer "the 70B server (job <jid>) reached health in N s; KV cache
  reported X blocks (`logs/vllm/llama3.1_70B-<jid>.out`)" over "70B works".
- **No check marks or cross glyphs** (no ✅, ✔, ✗, ✘, ❌). State outcomes in words —
  "Successful" / "Unsuccessful" / "Inconclusive" — or as a table column. Replace the
  existing `STATUS: GREEN ✅` banner and the `✗ Didn't work` sub-bullets in the guide
  with plain-language equivalents.
- Use **tables** for structured facts (cluster facts, the model→TP→memory plan, the
  job ledger below) and **bullet points** for procedures and findings. Reserve prose
  for rationale.
- Record **both successful and unsuccessful approaches.** A dead end (e.g. a TP size
  that OOM'd, a queue that rejected the job, a co-location attempt that collided on
  GPU 0) is as valuable to the reader as a success — state what you tried, what
  happened, the evidence, and the resolution. Keep the dated decisions log going.

**Required artifact — the job ledger.** Maintain a table mapping **every log file**
to its job and outcome, and keep it current as you submit new jobs. Seed it from the
existing logs (already inventoried below — note `logs/` is fully gitignored, so
enumerate with `ls logs/` and a depth-limited `find logs -maxdepth 2`, **not**
`git ls-files` and **not** a cluster-wide `find`):

| Log file(s) | Job id | Queue | Model / TP | Outcome | Root cause / note |
|---|---|---|---|---|---|
| `logs/probe_wrap_7186959.log`, `logs/probe_nvidia-smi_7186959.txt`, `logs/7186959.*.OU/.ER` | 7186959 | debug | Qwen2.5-0.5B / TP1 | Successful | Toolchain probe: torch saw the A100, vLLM 0.8.4 loaded the 0.5B on V1 and generated; `Exit_status=0` |
| `logs/vllm/qwen2.5_0.5B-7186960.wrap.log`, `logs/vllm/7186960.*.OU/.ER` | 7186960 | debug | Qwen2.5-0.5B / TP1 | Unsuccessful (aborted) | Orphaned two-job server; `qdel`'d because the dependent experiment job was rejected on debug (`max_run=1`/`queued=1`). Demonstrates the two-job pattern is invalid on debug |
| `logs/vllm/qwen2.5_0.5B-7186962.out`, `logs/paper/polaris_smoke_7186962.log`, `logs/paper/7186962.*.OU/.ER` | 7186962 | debug | Qwen2.5-0.5B / TP1 | Unsuccessful | vLLM came up (`Application startup complete`) but the health check hit the ALCF `http_proxy` and hung → `vLLM not healthy in 900s` (`Exit_status=45`). Fixed by the proxy bypass |
| `logs/vllm/qwen2.5_0.5B-7186966.out`, `logs/paper/polaris_smoke_7186966.log`, `logs/paper/7186966.*.OU/.ER` | 7186966 | debug | Qwen2.5-0.5B / TP1 | Successful | Full single-job orchestration: health in 21 s, discovery via HSN IP `10.201.3.1:8159`, `ping_servers` 0.55 s, 1 episode, `experiment_summary.csv` written; `Exit_status=0`. (Gameplay garbage = 0.5B model capacity, not pipeline) |
| `logs/build/decrypto_serve_venv.log`, `logs/build/runner_import_check.log`, `logs/build/vllm_help_check.log` | n/a (login) | — | — | Successful | One-time serving-venv build + import/`vllm --help` sanity checks |

Add a row for every job you submit (probes, the 70B isolation run, the 3-server
launch, the experiment), classify it Successful/Unsuccessful, and give the
one-line root cause anchored to the log line that proves it.

## Working method

- **Branch.** You are on `polaris-decrypto`. Stay on it (or branch from it).
- **Commit after every successful test rung.** As soon as a validation rung passes
  (a probe loads, a server reaches health, the matrix completes), `git add` the
  changed scripts/config/notes and commit with a message naming the job id and what
  it proved (e.g. `70B TP=4 server healthy, job 720xxxx; KV cache N blocks`). This
  gives you a clean checkpoint.
  - **On a failure, revert to the last good commit** rather than debugging on top of
    a broken tree: `git stash` or `git checkout -- <file>` to drop the bad change,
    or `git reset --hard <last-green-sha>` if you have diverged badly — then retry
    from known-good. Never let an unproven change sit uncommitted on top of a proven
    one; the ledger and the git history should agree on what is green.
  - Do not `git push` or open PRs unless the user asks.
- **Never `find` the cluster.** As stated in Objective 5: no wide `find` over
  `/lus/eagle`, `$HOME`, or `/`. Use `git grep` / `rg` / `git ls-files` in the repo
  and depth-limited `find` only under a specific small subdir.
- **Watch live, not the PBS spool.** `.OU`/`.ER` flush only at job end; every script
  mirrors to a live `…wrap.log` — `tail -f` that.
- **Prefer `*_polaris` copies over editing Midway scripts in place** so both ports
  stay diffable.
- When a cluster fact contradicts this prompt, trust the cluster and record the
  correction in the notes.

## Definition of done

- Three models staged under `$BASE/models/` and served concurrently at correct,
  *verified* TP sizes — smokes proven on `debug`, the concurrent multi-server run on
  `preemptable` (or your confirmed long multi-job queue).
- The 27-combo cross-play matrix passes a cheap 1-seed / 1-episode orchestration
  smoke, then the **full 8–16 h production run** completes and writes
  `results/polaris_3model/experiment_summary.csv` (results written incrementally so a
  preemption is recoverable); servers `qdel`'d on completion.
- `polaris_pbs_notes.md` + `polaris_setup_guide.md` updated to the register above:
  no check-mark glyphs, tables/bullets where appropriate, both successful and
  unsuccessful approaches recorded, and a complete **job ledger** mapping every log
  to a Successful/Unsuccessful outcome.
- No stale references remain (`agent_paths` Midway paths, dead doc links,
  future-tense scaling text) — verified with repo-scoped search, not a cluster `find`.
- Git history shows a commit at each green rung, and the tree matches the last
  green state (no unproven changes left dangling).
