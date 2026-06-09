#!/bin/bash
# Build the Decrypto serving venv on a Polaris LOGIN node, then pack it into one
# tarball for fast node-local-SSD staging in jobs (see polaris_pbs_notes.md).
#
# Why a venv + tarball (not a plain eagle venv):
#   eagle (Lustre) is fast for big sequential reads but catastrophic at the
#   ~70k-tiny-file metadata storm `import torch`/vllm triggers on a cold compute
#   node (MARSHAL measured ~19 min / effective hang). We therefore build once on
#   eagle, tar it, and every job extracts the ONE big file onto /local/scratch
#   (~12 s) and runs from there.
#
# Stack mirrors MARSHAL's PROVEN Polaris set (polaris_pbs_notes.md): cu124 is the
# native CUDA (12.4.1) and the GPUs are A100 sm_80, so torch 2.6.0+cu124 / vllm
# 0.8.4 are validated here. transformers is pinned <5 (tokenizers <0.22) to avoid
# the `all_special_tokens_extended` load error vLLM 0.8.x hits on transformers 5.
#
# One env hosts BOTH the vLLM server and the HTTP-only runner, so we add the
# Decrypto runner deps on top of the serving stack.
#
# Run:  bash slurm/build_decrypto_serve_venv.sh   (login node; needs internet)
set -euo pipefail

BASE=/lus/eagle/projects/lighthouse-uchicago/members/mehta5
VENV="$BASE/conda-envs/decrypto-serve"
TARBALL="$BASE/decrypto-serve-venv.tar"

export PIP_CACHE_DIR="$BASE/pip_cache"   # reuse MARSHAL's populated wheel cache
export TMPDIR="$BASE/tmp"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

echo "=== $(date): modules ==="
module use /soft/modulefiles
module load conda/2025-09-25
conda activate base
python --version

echo "=== $(date): create venv at $VENV ==="
rm -rf "$VENV"
python -m venv "$VENV"
# Activate by PATH (this is the build host == the venv's home path, so sourcing
# would also work, but we keep the same manual idiom the jobs use).
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
hash -r
python -m pip install --upgrade pip wheel setuptools

echo "=== $(date): serving stack (proven cu124 set) ==="
# torch first so vllm sees the cu124 build already satisfied.
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install vllm==0.8.4

echo "=== $(date): pin transformers<5 / tokenizers<0.22 / numpy<2 ==="
python -m pip install transformers==4.51.2 "tokenizers<0.22" "numpy<2.0"

echo "=== $(date): Decrypto runner deps (HTTP-only side) ==="
python -m pip install python-dotenv nltk gensim scipy pandas tqdm \
    anthropic openai requests "hydra-core==1.3.2" litellm omegaconf

echo "=== $(date): re-pin in case a runner dep moved transformers/numpy ==="
python -m pip install transformers==4.51.2 "tokenizers<0.22" "numpy<2.0"

echo "=== $(date): verify imports ==="
python - <<'PY'
import torch, vllm, transformers, tokenizers, numpy
import hydra, litellm, anthropic, gensim, openai, nltk, dotenv, omegaconf
print("torch       ", torch.__version__)
print("vllm        ", vllm.__version__)
print("transformers", transformers.__version__)
print("tokenizers  ", tokenizers.__version__)
print("numpy       ", numpy.__version__)
print("openai      ", openai.__version__)
print("ALL RUNNER + SERVING IMPORTS OK")
PY

echo "=== $(date): pack tarball -> $TARBALL ==="
# Archive the venv CONTENTS at the tar root (extracted with -C <dir>), matching
# the MARSHAL staging idiom: `tar xf $TARBALL -C $LOCAL_VENV`.
rm -f "$TARBALL"
tar cf "$TARBALL" -C "$VENV" .
echo "=== $(date): DONE. tarball: $(du -sh "$TARBALL" | cut -f1) ==="
