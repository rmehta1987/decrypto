#!/bin/bash
#SBATCH --job-name=bench
# SBATCH --nodes=2
# SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
# SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G
#SBATCH --constraint="a100|h100"
#SBATCH --partition=general
#SBATCH --output=/net/projects2/ycleong/sg/strategy-rl/decrypto/logs/paper/fig4_all_%j.out
#SBATCH --error=/net/projects2/ycleong/sg/strategy-rl/decrypto/logs/paper/fig4_all_%j.err
#SBATCH --export=ALL,HF_HOME=/net/projects2/ycleong/sg/strategy-rl/tmp/hf_cache,HUGGINGFACE_HUB_CACHE=/net/projects2/ycleong/sg/strategy-rl/tmp/hf_cache
#
# Resource directives (--nodes, --gpus-per-node) are set dynamically at
# submission time by the helper at the bottom of this script.  Run:
#
#   bash slurm/run_all.sbatch
#
# from the decrypto root directory to submit.
# N.B.: `sbatch slurm/run_all.sbatch` will not work.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# 1. Configuration — edit these arrays to match your experiment
# ──────────────────────────────────────────────────────────────────────

# Format: "model_key:hf_or_local_path:num_gpus"
TRAIN_PROTOCOL="generalist"
# TRAIN_PROTOCOL="tic_tac_toe"
# TRAIN_PROTOCOL="kuhn_poker"
# TRAIN_PROTOCOL="hanabi"
CKPT_STEP=199  # don't forget to update ckpt step in yaml config files as well!
models=(
    "llama3.1_70B:meta-llama/Meta-Llama-3.1-70B-Instruct:4"
    # "qwen3_4b:Qwen/Qwen3-4B:1"
    # "qwen3_4b_kuhn_poker:nics-efc/MARSHAL-Kuhn-Poker-Qwen3-4B:1"
    # "qwen3_4b_tic_tac_toe:nics-efc/MARSHAL-Tic-Tac-Toe-Qwen3-4B:1"
    # "qwen3_4b_hanabi:/net/projects2/ycleong/sg/strategy-rl/MARSHAL/results/hf_models/selfplay/hanabi_selfplay/checkpoint-${CKPT_STEP}:1"
    "qwen3_8b:Qwen/Qwen3-8B:2"
    "qwen3_8b_generalist:nics-efc/MARSHAL-Generalist-Qwen3-8B:2"
)

# How many GPUs each node has on your cluster (e.g. 4 for 4×A100, 8 for 8×H100).
# All models are bin-packed onto as few nodes as possible given this capacity.
NODE_GPU_CAPACITY=8

# CPUs allocated to each vLLM server (tokenization, HTTP serving)
CPUS_PER_MODEL=4

# CPUs allocated to the experiment runner (python run.py)
CPUS_PER_EXP=8

# Hydra config names to run sequentially (under config/paper/)
EXPERIMENT_CONFIGS=(
    # "figure_4_tom_piaget_test"
    # "figure_4_tom_piaget"
    "figure_4_tom_gopnik"
)

# EXP_NAME_SUFFIX="" # default
# EXP_NAME_SUFFIX="_qwen3_8b_generalist_${CKPT_STEP}"
EXP_NAME_SUFFIX="_qwen3_4b_${TRAIN_PROTOCOL}_${CKPT_STEP}_think2000"
# EXP_NAME_SUFFIX="_qwen3_4b_${TRAIN_PROTOCOL}_${CKPT_STEP}_only"

# ──────────────────────────────────────────────────────────────────────
# 2. Self-submitting wrapper — runs when executed outside SLURM
# ──────────────────────────────────────────────────────────────────────

if [ -z "${SLURM_JOB_ID:-}" ]; then
    cd /net/projects2/ycleong/sg/strategy-rl/decrypto

    total_gpus=0
    max_gpus=0
    for entry in "${models[@]}"; do
        g="${entry##*:}"
        total_gpus=$((total_gpus + g))
        (( g > max_gpus )) && max_gpus=$g
    done

    if [ "$NODE_GPU_CAPACITY" -lt "$max_gpus" ]; then
        echo "ERROR: NODE_GPU_CAPACITY=$NODE_GPU_CAPACITY but largest model needs $max_gpus GPUs"
        exit 1
    fi

    # compute number of nodes needed
    if [ "$total_gpus" -le "$NODE_GPU_CAPACITY" ]; then
        num_nodes=1
        gpus_per_node=$total_gpus
    else
        gpus_per_node=$NODE_GPU_CAPACITY
        num_nodes=$(( (total_gpus + gpus_per_node - 1) / gpus_per_node ))
    fi

    exp_tasks=$(( (CPUS_PER_EXP + CPUS_PER_MODEL - 1) / CPUS_PER_MODEL ))
    ntasks=$(( ${#models[@]} + exp_tasks ))

    echo "Models require $total_gpus GPUs total (max $max_gpus per model)"
    echo "Requesting $num_nodes node(s) × $gpus_per_node GPUs, $ntasks tasks × $CPUS_PER_MODEL CPUs"

    sbatch \
        --nodes="$num_nodes" \
        --gpus-per-node="$gpus_per_node" \
        --ntasks="$ntasks" \
        --cpus-per-task="$CPUS_PER_MODEL" \
        "$0"
    exit 0
fi

# ──────────────────────────────────────────────────────────────────────
# 3. Inside SLURM — launch vLLM servers and run experiments
# ──────────────────────────────────────────────────────────────────────

# Activate the conda environment
source /net/projects2/ycleong/sg/miniconda3/etc/profile.d/conda.sh
conda activate decrypto

cd /net/projects2/ycleong/sg/strategy-rl/decrypto

EXPECTED_MODELS=${#models[@]}
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-1}

echo "Job $SLURM_JOB_ID on nodes: ${NODES[*]} ($GPUS_PER_NODE GPUs each)"
echo "Launching $EXPECTED_MODELS vLLM servers..."

# Track how many GPUs are already claimed on each node
declare -A node_gpus_used
for node in "${NODES[@]}"; do
    node_gpus_used[$node]=0
done

# Build the servers JSON file and launch each model
SERVERS_FILE="/tmp/decrypto_servers_${SLURM_JOB_ID}.json"
echo '[' > "$SERVERS_FILE"
PORT=8000
FIRST=true  # track first model to add comma separator

for entry in "${models[@]}"; do
    model_key="${entry%%:*}"
    num_gpus="${entry##*:}"
    model_path="${entry#*:}"
    model_path="${model_path%:*}"

    # First-fit bin-packing: find a node with enough free GPUs
    assigned_node=""
    for node in "${NODES[@]}"; do
        used=${node_gpus_used[$node]}
        avail=$((GPUS_PER_NODE - used))
        if [ "$avail" -ge "$num_gpus" ]; then
            assigned_node="$node"
            node_gpus_used[$node]=$((used + num_gpus))
            break
        fi
    done

    if [ -z "$assigned_node" ]; then
        echo "FATAL: Cannot place $model_key ($num_gpus GPUs) — no node has enough free GPUs."
        echo "  Nodes: ${NODES[*]} ($GPUS_PER_NODE GPUs each), used so far: ${node_gpus_used[*]}"
        exit 1
    fi

    port=$PORT
    PORT=$((PORT + 1)) # prevent port conflicts

    echo "  $model_key  →  $assigned_node:$port  ($num_gpus GPUs)"

    srun --nodes=1 --ntasks=1 --gpus="$num_gpus" \
         --cpus-per-task="$CPUS_PER_MODEL" \
         --nodelist="$assigned_node" --exact \
         vllm serve "$model_path" \
             --port "$port" \
             --enable-prefix-caching \
             --tensor-parallel-size "$num_gpus" \
             --trust-remote-code \
             --disable-sliding-window \
             --disable-log-stats &

    # Append entry to servers JSON
    $FIRST && FIRST=false || echo ',' >> "$SERVERS_FILE"
    cat >> "$SERVERS_FILE" <<ENTRY
  {"model_key": "$model_key", "model_id": "$model_path", "urls": ["http://${assigned_node}:${port}/v1"], "job_ids": ["$SLURM_JOB_ID"]}
ENTRY
done

echo ']' >> "$SERVERS_FILE"
export DECRYPTO_SERVERS_FILE="$SERVERS_FILE"

echo ""
echo "Servers file: $SERVERS_FILE"
cat "$SERVERS_FILE"
echo ""

# ──────────────────────────────────────────────────────────────────────
# 4. Wait for every vLLM server to become healthy
# ──────────────────────────────────────────────────────────────────────

echo "Waiting for vLLM models to load into VRAM..."
while true; do
    READY_COUNT=$(python -m slurm.ping_servers 2>/dev/null | grep -c "Reply time" || true)
    if [ "$READY_COUNT" -ge "$EXPECTED_MODELS" ]; then
        echo "All $EXPECTED_MODELS servers are ready!"
        break
    fi
    echo "  $READY_COUNT / $EXPECTED_MODELS servers ready — sleeping 30s..."
    sleep 30
done

# ──────────────────────────────────────────────────────────────────────
# 5. Run experiments
# ──────────────────────────────────────────────────────────────────────

for config in "${EXPERIMENT_CONFIGS[@]}"; do
    echo ""
    echo "===== Running experiment: $config ====="
    python run.py \
        --config-path=config/paper \
        --config-name="$config" \
        +get_models_from_slurm=true \
        exp_name=${config}${EXP_NAME_SUFFIX}
done

echo ""
echo "All experiments finished."
rm -f "$SERVERS_FILE"
