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
# queue that allows >=2 concurrent jobs (e.g. prod, or inside a reservation).
#
# Run from the repo root:  bash slurm/launch_servers_polaris.sh
set -euo pipefail

BASE=/lus/eagle/projects/lighthouse-uchicago/members/mehta5
REPO_ROOT="$BASE/decrypto"
VENV_TARBALL="${VENV_TARBALL:-$BASE/decrypto-serve-venv.tar}"

cd "$REPO_ROOT"
mkdir -p "$REPO_ROOT/logs/vllm" "$REPO_ROOT/logs/paper"

# Format: "model_key:model_path"  (model_path is a LOCAL path — compute nodes are
# offline; pre-stage models on a login node into $BASE/models). ngpus = TP size.
models=(
    "qwen2.5_0.5B:$BASE/models/Qwen2.5-0.5B-Instruct"
)
ngpus=(1)   # TP=1: 0.5B fits one 40 GB A100; prove the pipeline, then scale.

EXPECTED_MODELS=${#models[@]}
if [ ${#ngpus[@]} -ne "$EXPECTED_MODELS" ]; then
  echo "ngpus / models length mismatch"; exit 1
fi

if [ ! -f "$VENV_TARBALL" ]; then
  echo "venv tarball missing: $VENV_TARBALL (build it: bash slurm/build_decrypto_serve_venv.sh)"; exit 1
fi

# Fresh per-launch discovery dir so a previous run's stale JSON never leaks in.
LAUNCH_ID="$(date +%Y%m%d_%H%M%S)_$$"
SERVERS_DIR="$REPO_ROOT/logs/servers/$LAUNCH_ID"
mkdir -p "$SERVERS_DIR"
echo "SERVERS_DIR=$SERVERS_DIR"

SERVER_JIDS=""
for i in "${!models[@]}"; do
  entry="${models[$i]}"
  gpu_count="${ngpus[$i]}"
  model_key="${entry%%:*}"
  model_path="${entry#*:}"
  port=$((8000 + RANDOM % 1000))

  jid=$(
    qsub \
      -v MODEL_KEY="$model_key",MODEL_PATH="$model_path",PORT="$port",TP="$gpu_count",SERVERS_DIR="$SERVERS_DIR",VENV_TARBALL="$VENV_TARBALL" \
      slurm/server_vllm_polaris.pbs
  )
  echo "Submitted server ${model_key} (TP=${gpu_count}) on port ${port}: jid=${jid}"
  SERVER_JIDS="${SERVER_JIDS:+${SERVER_JIDS}:}${jid}"
done

echo "Submitting experiment job (depends on servers starting)..."
if ! exp_jid=$(
  qsub \
    -W depend=after:"$SERVER_JIDS" \
    -v SERVER_JIDS="$SERVER_JIDS",EXPECTED_MODELS="$EXPECTED_MODELS",DECRYPTO_SERVERS_FILE="$SERVERS_DIR" \
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
echo "Watch:  tail -f $REPO_ROOT/logs/vllm/${models[0]%%:*}-<jid>.wrap.log"
echo "        tail -f $REPO_ROOT/logs/paper/polaris_smoke_<jid>.log"
echo "Results: $REPO_ROOT/results/polaris_smoke/experiment_summary.csv"
