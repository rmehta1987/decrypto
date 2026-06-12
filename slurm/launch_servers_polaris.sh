#!/bin/bash
# Polaris (PBS Pro) launcher for the Decrypto self-hosted-vLLM pipeline.
# PBS analog of slurm/launch_servers_midway.sh — the MULTI-NODE / multi-model
# design: separate server job(s) + a dependent experiment job.
#
#   1. Mints a fresh per-launch SERVERS_DIR (the DECRYPTO_SERVERS_FILE the runner
#      reads — a directory of <jobid>.json files the server jobs write once up).
#   2. `qsub`s one server_vllm_polaris.pbs per model (PBS has no `--wrap`; params
#      go via `qsub -v`). Captures each full PBS job id.
#   3. `qsub`s run_exp_polaris.pbs as a DEPENDENT job (`-W depend=after:<jids>` —
#      PBS `after` = start once the servers have STARTED, matching Midway's
#      `--dependency=after`). It waits for health, runs the games, then qdels the
#      servers.
#
# ⚠️ THE `debug` QUEUE CANNOT RUN THIS. debug enforces max_run=1 AND
# queued_jobs_threshold=1 per user (`qstat -Qf debug`), so the server and
# experiment jobs can never be Running at the same time — and since the
# experiment qdels the server, two separate jobs would deadlock. On debug use the
# single-job  `qsub slurm/smoke_polaris.pbs`  instead. Use THIS launcher only on a
# queue that allows >=2 concurrent jobs — `preemptable` (1-10 nodes, <=72 h) is
# the verified choice; `demand` (<=1 h, by request) and `prod` (>=10 nodes/job)
# do not fit the long 1-node-per-job pattern.
#
# Run from the repo root:  bash slurm/launch_servers_polaris.sh
# Scale/queue knobs via env, e.g.:
#   SERVER_WALLTIME=17:00:00 EXP_WALLTIME=16:00:00 SEEDS="0 1 2 3 4" \
#     bash slurm/launch_servers_polaris.sh
set -euo pipefail

BASE=/lus/eagle/projects/lighthouse-uchicago/members/mehta5
REPO_ROOT="$BASE/decrypto"
VENV_TARBALL="${VENV_TARBALL:-$BASE/decrypto-serve-venv.tar}"

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs/vllm" "$REPO_ROOT/logs/paper"

# ── Queue + walltime + scale knobs (override via env at launch time) ──
# debug cannot host this multi-job pattern (max_run=1/user); `preemptable`
# (1-10 nodes, <=72 h, 20 queued/user) is the working long-run queue. Jobs
# there can be preempted, so submit with `-r y` (rerun) and write results
# incrementally. Keep SERVER_WALLTIME >= EXP_WALLTIME so no server dies
# mid-game.
QUEUE="${QUEUE:-preemptable}"
SERVER_WALLTIME="${SERVER_WALLTIME:-02:00:00}"
EXP_WALLTIME="${EXP_WALLTIME:-01:30:00}"
CONFIG_NAME="${CONFIG_NAME:-local_polaris_3model}"
EXP_NAME="${EXP_NAME:-polaris_3model}"
SEEDS="${SEEDS:-}"                 # e.g. SEEDS="0 1 2 3 4" (space-separated)
NUM_EPISODES="${NUM_EPISODES:-}"
# 70B cold load reads ~141 GB off Lustre: give the server's own health wait and
# the experiment's ready-poll far more than the 0.5B's 21 s.
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-3600}"

# Format: "model_key:model_path"  (model_path is a LOCAL path — compute nodes are
# offline; pre-stage models on a login node into $BASE/models). ngpus = TP size:
# it must divide both num_attention_heads AND num_key_value_heads (GQA) of the
# model, and weights/TP must fit a 40 GB A100 with KV headroom.
#   Llama-3.1-70B: bf16 weights ~131 GiB -> TP=4 (~33 GiB/GPU; the only
#                  single-node fit on 40 GB cards); heads 64/8, both %4==0.
#   Qwen3-8B/-4B:  fit a single card -> TP=1 (heads 32/8).
# One model per node-job (simplest correct layout; co-locating the two small
# models on one node would need CUDA_VISIBLE_DEVICES pinning per server).
models=(
    "llama3.1_70B:$BASE/models/Meta-Llama-3.1-70B-Instruct"
    "qwen3_8b:$BASE/models/Qwen3-8B"
    "qwen3_4b:$BASE/models/Qwen3-4B"
)
ngpus=(4 1 1)
# Per-model engine knobs: 70B at TP=4 is weight-tight on 40 GB cards, so give
# it a higher mem fraction; small models keep the proven defaults.
mem_utils=(0.92 0.90 0.90)
max_lens=(8192 8192 8192)

EXPECTED_MODELS=${#models[@]}
if [ ${#ngpus[@]} -ne "$EXPECTED_MODELS" ] || [ ${#mem_utils[@]} -ne "$EXPECTED_MODELS" ] || [ ${#max_lens[@]} -ne "$EXPECTED_MODELS" ]; then
  echo "ngpus / mem_utils / max_lens / models length mismatch"; exit 1
fi

for i in "${!models[@]}"; do
  p="${models[$i]#*:}"
  if [ ! -d "$p" ]; then echo "model path missing (pre-stage it on a login node): $p"; exit 1; fi
done

if [ ! -f "$VENV_TARBALL" ]; then
  echo "venv tarball missing: $VENV_TARBALL (build it: bash slurm/build_decrypto_serve_venv.sh)"; exit 1
fi

# Fresh per-launch discovery dir so a previous run's stale JSON never leaks in.
LAUNCH_ID="$(date +%Y%m%d_%H%M%S)_$$"
SERVERS_DIR="$REPO_ROOT/logs/servers/$LAUNCH_ID"
mkdir -p "$SERVERS_DIR"
echo "SERVERS_DIR=$SERVERS_DIR"

echo "queue=$QUEUE server_walltime=$SERVER_WALLTIME exp_walltime=$EXP_WALLTIME config=$CONFIG_NAME exp_name=$EXP_NAME"

SERVER_JIDS=""
for i in "${!models[@]}"; do
  entry="${models[$i]}"
  gpu_count="${ngpus[$i]}"
  mem_util="${mem_utils[$i]}"
  max_len="${max_lens[$i]}"
  model_key="${entry%%:*}"
  model_path="${entry#*:}"
  port=$((8000 + RANDOM % 1000))

  jid=$(
    qsub \
      -q "$QUEUE" -r y -l walltime="$SERVER_WALLTIME" \
      -v MODEL_KEY="$model_key",MODEL_PATH="$model_path",PORT="$port",TP="$gpu_count",SERVERS_DIR="$SERVERS_DIR",VENV_TARBALL="$VENV_TARBALL",GPU_MEM_UTIL="$mem_util",MAX_MODEL_LEN="$max_len",HEALTH_TIMEOUT="$HEALTH_TIMEOUT" \
      slurm/server_vllm_polaris.pbs
  )
  echo "Submitted server ${model_key} (TP=${gpu_count} mem_util=${mem_util} max_len=${max_len}) on port ${port}: jid=${jid}"
  SERVER_JIDS="${SERVER_JIDS:+${SERVER_JIDS}:}${jid}"
done

echo "Submitting experiment job (depends on servers starting)..."
if ! exp_jid=$(
  qsub \
    -q "$QUEUE" -r y -l walltime="$EXP_WALLTIME" \
    -W depend=after:"$SERVER_JIDS" \
    -v SERVER_JIDS="$SERVER_JIDS",EXPECTED_MODELS="$EXPECTED_MODELS",DECRYPTO_SERVERS_FILE="$SERVERS_DIR",CONFIG_NAME="$CONFIG_NAME",EXP_NAME="$EXP_NAME",WAIT_TIMEOUT="$WAIT_TIMEOUT",SEEDS="$SEEDS",NUM_EPISODES="$NUM_EPISODES" \
    slurm/run_exp_polaris.pbs
); then
  echo "ERROR: could not submit the dependent experiment job (above)."
  echo "       On the 'debug' queue this is expected (max_run=1 / queued=1):"
  echo "       use the single-job  qsub slurm/smoke_polaris.pbs  instead."
  echo "       Cancelling the server job(s) so they don't run orphaned: ${SERVER_JIDS//:/ }"
  qdel ${SERVER_JIDS//:/ } 2>/dev/null || true
  exit 1
fi
echo "Submitted experiment job: jid=$exp_jid"
echo
echo "Watch:  tail -f $REPO_ROOT/logs/vllm/<model_key>-<jid>.wrap.log   (one per server)"
echo "        tail -f $REPO_ROOT/logs/paper/polaris_smoke_<jid>.log"
echo "Results: $REPO_ROOT/results/$EXP_NAME/experiment_summary.csv"
