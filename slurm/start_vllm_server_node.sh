#!/bin/bash
# Per-NODE vLLM server bootstrap for Polaris — the body of
# server_vllm_polaris.pbs extracted so a fused multi-node job can launch one
# server per node via `mpiexec --hosts <node>`. Runs entirely on the target
# node; stages the venv to node-local scratch, starts vLLM, waits for health,
# registers {model_key, model_id, urls, job_ids} into SERVERS_DIR, then waits
# on the vLLM process until killed.
#
# Args (positional):
#   1 MODEL_KEY  2 MODEL_PATH  3 TP  4 PORT  5 SERVERS_DIR
#   6 GPU_MEM_UTIL  7 MAX_MODEL_LEN  8 MNBT(''=default)  9 MNSEQS(''=default)
#   10 HEALTH_TIMEOUT  11 JOBTAG (bare job id of the enclosing PBS job)
set -uo pipefail

MODEL_KEY="$1"; MODEL_PATH="$2"; TP="$3"; PORT="$4"; SERVERS_DIR="$5"
GPU_MEM_UTIL="$6"; MAX_MODEL_LEN="$7"; MNBT="$8"; MNSEQS="$9"
HEALTH_TIMEOUT="${10}"; JOBTAG="${11}"

BASE=/lus/eagle/projects/lighthouse-uchicago/members/mehta5
REPO_ROOT="$BASE/decrypto"
VENV_TARBALL="${VENV_TARBALL:-$BASE/decrypto-serve-venv.tar}"

WRAP_LOG="$REPO_ROOT/logs/vllm/${MODEL_KEY}-${JOBTAG}.wrap.log"
exec > >(stdbuf -oL -eL tee -a "$WRAP_LOG") 2>&1
echo "=== fused-node server $MODEL_KEY (job $JOBTAG) on $(hostname) at $(date) ==="

module use /soft/modulefiles
module load conda/2025-09-25
conda activate base
unset TMPDIR
NODE_SCRATCH=/local/scratch
[ -d "$NODE_SCRATCH" ] || NODE_SCRATCH=/tmp
export TMPDIR="$NODE_SCRATCH/${USER}_${JOBTAG}_${MODEL_KEY}"
mkdir -p "$TMPDIR"

if [ ! -f "$VENV_TARBALL" ]; then echo "FATAL: venv tarball not found: $VENV_TARBALL"; exit 2; fi
LOCAL_VENV="$TMPDIR/decrypto-serve"
mkdir -p "$LOCAL_VENV"
_t0=$SECONDS
tar xf "$VENV_TARBALL" -C "$LOCAL_VENV"
echo "venv extracted in $((SECONDS-_t0))s"
export VIRTUAL_ENV="$LOCAL_VENV"
export PATH="$LOCAL_VENV/bin:$PATH"
hash -r

export HF_HOME="$BASE/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="$BASE/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$BASE/torchinductor_cache"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
export no_proxy="localhost,127.0.0.1,::1,10.201.0.0/16,.hsn.cm.polaris.alcf.anl.gov,.alcf.anl.gov"
export NO_PROXY="$no_proxy"

HSN_FQDN="$(hostname).hsn.cm.polaris.alcf.anl.gov"
HSN_IP="$(getent hosts "$HSN_FQDN" 2>/dev/null | awk '{print $1; exit}')"
SERVER_ADDR="${HSN_IP:-$HSN_FQDN}"
echo "HSN address for discovery: $SERVER_ADDR"

SERVERS_FILE="$SERVERS_DIR/${JOBTAG}_${MODEL_KEY}.json"

cleanup() {
    echo "$(date): node-server cleanup ($MODEL_KEY)"
    rm -f "$SERVERS_FILE" "$SERVERS_FILE.tmp" 2>/dev/null || true
    [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

EXTRA_ARGS=()
[ -n "$MNBT" ] && EXTRA_ARGS+=(--max-num-batched-tokens "$MNBT")
[ -n "$MNSEQS" ] && EXTRA_ARGS+=(--max-num-seqs "$MNSEQS")
echo ">>> launching vLLM ($MODEL_KEY tp=$TP port=$PORT mem=$GPU_MEM_UTIL len=$MAX_MODEL_LEN mnbt=${MNBT:-default} mnseqs=${MNSEQS:-default})"
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --trust-remote-code \
    --disable-log-stats \
    "${EXTRA_ARGS[@]}" &
VLLM_PID=$!

health_ok() {
    curl -sf --noproxy '*' "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    python - "$PORT" <<'PYEOF'
import sys, urllib.request
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    op.open(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=3)
    sys.exit(0)
except Exception:
    sys.exit(1)
PYEOF
}

echo ">>> waiting for vLLM health on :$PORT (timeout ${HEALTH_TIMEOUT}s)"
_t0=$SECONDS
until health_ok; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "FATAL: vLLM ($MODEL_KEY) exited before becoming healthy"; exit 44
    fi
    if [ $((SECONDS-_t0)) -gt "$HEALTH_TIMEOUT" ]; then
        echo "FATAL: vLLM ($MODEL_KEY) not healthy within ${HEALTH_TIMEOUT}s"; exit 45
    fi
    sleep 5
done
echo ">>> vLLM ($MODEL_KEY) healthy after $((SECONDS-_t0))s"
echo ">>> KV-cache evidence:"
grep -E "KV cache size|Maximum concurrency" "$WRAP_LOG" | tail -2 || true

python - "$MODEL_KEY" "$MODEL_PATH" "$SERVER_ADDR" "$PORT" "$JOBTAG" "$SERVERS_FILE" <<'PYEOF'
import json, os, sys
mk, mid, addr, port, jid, out = sys.argv[1:7]
entry = {"model_key": mk, "model_id": mid, "urls": [f"http://{addr}:{port}/v1"], "job_ids": [jid]}
tmp = out + ".tmp"
json.dump(entry, open(tmp, "w")); os.replace(tmp, out)
print("registered:", entry)
PYEOF

echo ">>> server $MODEL_KEY READY on http://$SERVER_ADDR:$PORT/v1"
wait "$VLLM_PID"
