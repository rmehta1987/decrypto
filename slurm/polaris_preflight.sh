# Polaris per-node pre-flight checks, SOURCED (not executed) by the Polaris job
# scripts after the venv tarball is extracted. Each check prints what it verified
# and returns non-zero on failure; the caller decides the exit code.
#
# Why these exist: a missing path on Polaris tends to change behaviour silently
# instead of failing (MARSHAL/pedramh: a nonexistent LD_LIBRARY_PATH entry is
# ignored and a different transport loads). Check up front, fail loudly.

# The relocated venv's bin/python is a symlink into the conda install tree it was
# built from (/soft/applications/conda/2025-09-25/mconda3/bin/python). No conda
# module is loaded at run time, so nothing else would notice if ALCF moved that
# tree -- every tarball venv would break at once.
decrypto_venv_preflight() {  # $1 = extracted venv dir
    local venv="$1" real
    real="$(readlink -f "$venv/bin/python" 2>/dev/null)"
    if [ -z "$real" ] || [ ! -x "$real" ]; then
        echo "FATAL: venv interpreter does not resolve: $venv/bin/python -> $(readlink "$venv/bin/python" 2>/dev/null || echo '<no link>')"
        echo "       The conda tree it was built on is gone or moved; rebuild: bash slurm/build_decrypto_serve_venv.sh"
        return 1
    fi
    echo "venv interpreter: $venv/bin/python -> $real"
}

# A local model path that doesn't exist is not rejected by vLLM up front: it is
# taken as a Hub repo id and fails only later under HF_HUB_OFFLINE, after the
# venv is staged (and, in a fused job, after the other servers spent their load
# time). Check each path before anything is launched.
decrypto_model_path_check() {  # $1 = local model dir
    if [ ! -f "$1/config.json" ]; then
        echo "FATAL: model path missing or has no config.json: $1 (pre-stage it on a login node)"
        return 1
    fi
}

# The comm stack a server actually runs with, for its log header: the NCCL torch
# loaded (version + the .so it mapped), every NCCL_*/FI_* variable, and
# LD_LIBRARY_PATH. This is the baseline to diff against if a model ever spans
# nodes (HANDOFF_POLARIS_NCCL_ENV_2026-09-28.md §3C). Informational only: it
# never fails the job. Call after the venv's python is first on PATH.
decrypto_comm_stack_header() {
    local vars
    echo "--- comm stack on $(hostname) ---"
    python - <<'PY' || echo "  WARNING: could not query torch's NCCL (continuing)"
import torch
v = torch.cuda.nccl.version()
lib = next((l.split()[-1] for l in open("/proc/self/maps") if "libnccl" in l), "<libnccl not mapped>")
print("  torch", torch.__version__, "| NCCL", ".".join(map(str, v)) if isinstance(v, tuple) else v, "|", lib)
PY
    vars="$(env | grep -E '^(NCCL|FI)_' | sort)"
    if [ -n "$vars" ]; then echo "$vars" | sed 's/^/  /'; else echo "  NCCL_*/FI_*: (none set)"; fi
    echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
    echo "---"
}
