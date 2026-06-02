#!/bin/bash
# Midway-flavored launcher for the Decrypto smoke test.
# - Submits one vLLM server per entry in `models`, on the `test` partition
#   with an H200 constraint.
# - Then submits run_exp_midway.sbatch as a dependent job.
# Differs from launch_servers.sh: paths, account/partition/constraint,
# and the vllm command is wrapped with mamba activation of the vllm-probe env.

set -euo pipefail

REPO_ROOT=/project/rcc/mehta5/decrypto
ENV_PATH=/project/rcc/mehta5/conda-envs/vllm-probe
HF_CACHE=/project/rcc/mehta5/hf_cache
INDUCTOR_CACHE=/project/rcc/mehta5/torchinductor_cache

mkdir -p "$REPO_ROOT/logs/vllm" "$HF_CACHE" "$INDUCTOR_CACHE"

# Format: "model_key:model_path_or_hf_id"
models=(
    "llama3.1_70B:/project/rcc/mehta5/vllm/models/Meta-Llama-3.1-70B-Instruct"
)
ngpus=(4)  # TP=4: 70B at bf16 is ~140 GB, spread across whole 4× H200 node

EXPECTED_MODELS=${#models[@]}
if [ ${#ngpus[@]} -ne $EXPECTED_MODELS ]; then
  echo "ngpus / models length mismatch"; exit 1
fi

SERVER_JIDS=""
for i in "${!models[@]}"; do
  entry="${models[$i]}"
  gpu_count="${ngpus[$i]}"
  model_key="${entry%%:*}"
  model_path="${entry#*:}"
  port=$((8000 + RANDOM % 1000))

  jid=$(
    sbatch \
      --parsable \
      --account=rcc-staff \
      --partition=test \
      --constraint=H200 \
      --gres=gpu:${gpu_count} \
      --cpus-per-task=16 \
      --mem=128G \
      --time=02:00:00 \
      --nodes=1 \
      --ntasks=1 \
      --output "${REPO_ROOT}/logs/vllm/${model_key}-%j.out" \
      --error  "${REPO_ROOT}/logs/vllm/${model_key}-%j.err" \
      --export=ALL,HF_HOME=${HF_CACHE},HUGGINGFACE_HUB_CACHE=${HF_CACHE},TORCHINDUCTOR_CACHE_DIR=${INDUCTOR_CACHE} \
      --job-name "${model_key}:${port}" \
      --wrap "$(cat <<EOF
set -euo pipefail
# Clear any inherited TMPDIR / SLURM scratch path from the submitting shell —
# a stale value from a previous (now-cancelled) job will cause torch.compile's
# autotune cache to try to write into a /scratch/local/jobs/<old_jid> path
# that no longer exists, killing the engine during model load.
unset TMPDIR SLURM_TMPDIR
export TMPDIR=/tmp/\${USER}_\${SLURM_JOB_ID}
mkdir -p "\$TMPDIR"

module load python/miniforge-25.3.0
eval "\$(mamba shell hook --shell bash)"
mamba activate ${ENV_PATH}
vllm serve ${model_path} \
  --port ${port} \
  --enable-prefix-caching \
  --enforce-eager \
  --tensor-parallel-size ${gpu_count} \
  --trust-remote-code \
  --disable-sliding-window \
  --disable-log-stats
EOF
)"
  )

  echo "Submitted ${model_key} on port ${port}: jid=${jid}"
  SERVER_JIDS="${SERVER_JIDS:+${SERVER_JIDS}:}${jid}"
done

echo "Submitting evaluation job (depends on servers starting)..."
cd "$REPO_ROOT"
sbatch \
  --dependency=after:$SERVER_JIDS \
  --export=ALL,SERVER_JIDS=$SERVER_JIDS,EXPECTED_MODELS=$EXPECTED_MODELS \
  slurm/run_exp_midway.sbatch
