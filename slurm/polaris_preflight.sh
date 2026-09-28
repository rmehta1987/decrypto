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
