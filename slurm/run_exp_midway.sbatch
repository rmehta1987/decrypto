#!/bin/bash
#SBATCH --job-name=decrypto_midway_smoke
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/project/rcc/mehta5/decrypto/logs/paper/midway_smoke_%j.out
#SBATCH --error=/project/rcc/mehta5/decrypto/logs/paper/midway_smoke_%j.err

set -euo pipefail

REPO_ROOT=/project/rcc/mehta5/decrypto
ENV_PATH=/project/rcc/mehta5/conda-envs/vllm-probe

# Drop any stale TMPDIR inherited from a previous job (see launcher for context).
unset TMPDIR SLURM_TMPDIR
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

mkdir -p "$REPO_ROOT/logs/paper"
cd "$REPO_ROOT"

module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate "$ENV_PATH"

EXPECTED_MODELS=${EXPECTED_MODELS:-1}

echo "Waiting for vLLM models to load into VRAM..."
while true; do
  READY_COUNT=$(python -m slurm.ping_servers 2>/dev/null | grep -c "Reply time" || true)
  if [ "$READY_COUNT" -ge "$EXPECTED_MODELS" ]; then
    echo "All $EXPECTED_MODELS servers ready."
    break
  fi
  echo "  $READY_COUNT / $EXPECTED_MODELS ready — sleeping 30s..."
  sleep 30
done

python run.py \
  --config-path=config/examples \
  --config-name=local_midway \
  +get_models_from_slurm=true \
  exp_name=midway_smoke

if [ -n "${SERVER_JIDS:-}" ]; then
  echo "Smoke test finished. Cancelling vLLM server jobs to free the GPU."
  scancel ${SERVER_JIDS//:/ }
fi
