#!/bin/bash
#SBATCH --job-name=vllm013-probe
#SBATCH --account=rcc-staff
#SBATCH --partition=test
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/project/rcc/mehta5/decrypto/logs/vllm013-probe-%j.log

# Probe: does vLLM 0.13.0 + torch 2.9.0 (repo's pinned versions, built against
# CUDA 12.8) actually run on this cluster's driver 535 (CUDA 12.2 max) via
# CUDA Minor Version Compatibility?
#
# Pre-req: env at /project/rcc/mehta5/conda-envs/vllm-probe already created
# on the build partition (see project-context for setup commands).

set -euo pipefail

module load python/miniforge-25.3.0
source activate /project/rcc/mehta5/conda-envs/vllm-probe

echo "=== Stage 0: nvidia-smi ==="
nvidia-smi
echo

echo "=== Stage 1: versions ==="
python - <<'PY'
import sys, torch, vllm
print("python:", sys.version.split()[0])
print("torch :", torch.__version__, "(built against CUDA", torch.version.cuda, ")")
print("vllm  :", vllm.__version__)
PY
echo

echo "=== Stage 2: torch can see the GPU ==="
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device 0      :", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
print("matmul ok     :", (x @ x).sum().item() != 0)
PY
echo

echo "=== Stage 3: vLLM can load + generate ==="
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(
    model="/project/rcc/mehta5/vllm/models/Qwen2.5-0.5B-Instruct",
    gpu_memory_utilization=0.3,
    max_model_len=2048,
    enforce_eager=True,   # skip CUDA graph capture to isolate basic op failures
)
out = llm.generate(
    ["Say hi in five words."],
    SamplingParams(max_tokens=20, temperature=0.0),
)
print("OUT:", out[0].outputs[0].text)
PY
echo

echo "=== Probe complete ==="
