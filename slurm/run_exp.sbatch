#!/bin/bash
#SBATCH --job-name=decrypto_piaget_claude
#SBATCH --output=logs/paper/fig4_%j.out
#SBATCH --error=logs/paper/fig4_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

cd /net/projects2/ycleong/sg/strategy-rl/decrypto

source /net/projects2/ycleong/sg/miniconda3/etc/profile.d/conda.sh
conda activate decrypto

# 1. Wait for the model weights to load into VRAM (takes ~1-3 mins)
EXPECTED_MODELS=${EXPECTED_MODELS:-0}
# EXPECTED_MODELS=2
echo "Waiting for vLLM models to load into VRAM..."
while true; do
  # ping_servers.py prints "Reply time: X sec" for every healthy server
  READY_COUNT=$(python -m slurm.ping_servers | grep -c "Reply time")
  if [ "$READY_COUNT" -ge "$EXPECTED_MODELS" ]; then
      echo "All $EXPECTED_MODELS servers are ready!"
      break
  fi
  echo "$READY_COUNT out of $EXPECTED_MODELS servers ready. Sleeping 30s..."
  sleep 30
done

# 2. Run the experiments
python run.py \
    --config-path=config/paper \
    --config-name=figure_4_tom_piaget \
    +get_models_from_slurm=true \
    exp_name=figure_4_tom_piaget_qwen3_4b
# python run.py \
#     --config-path=config/paper \
#     --config-name=figure_4_tom_gopnik \
#     +get_models_from_slurm=true \
#     exp_name=figure_4_tom_gopnik_qwen3_4b

# # 3. Clean up: Kill the vLLM servers to save GPU quota
# if [ -n "$SERVER_JIDS" ]; then
#   echo "Experiments finished. Shutting down vLLM servers to save quota..."
#   # Replace the colons with spaces and cancel the jobs
#   scancel ${SERVER_JIDS//:/ }
# fi
