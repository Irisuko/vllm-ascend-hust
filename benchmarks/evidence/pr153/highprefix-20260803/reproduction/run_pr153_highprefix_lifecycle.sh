#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 <native|mapped> <lifecycle-index> <run-dir> [container]" >&2
    exit 2
fi

MODE=$1
LIFECYCLE_INDEX=$2
RUN_DIR=$3
CONTAINER=${4:-source-dev-pr153-highprefix-npu4}
PHYSICAL_DEVICE_ID=${PHYSICAL_DEVICE_ID:-4}
PORT=${PORT:-8081}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}
MODEL_PATH=${MODEL_PATH:-/workspace/model}
DEVICE_KV_BYTES=${DEVICE_KV_BYTES:-268435456}
CPU_KV_BYTES=${CPU_KV_BYTES:-1073741824}
REQUEST_COUNT=${REQUEST_COUNT:-200}
REQUEST_RATE=${REQUEST_RATE:-1.0}
PREFIX_TOKENS=${PREFIX_TOKENS:-3840}
SUFFIX_TOKENS=${SUFFIX_TOKENS:-256}
OUTPUT_TOKENS=${OUTPUT_TOKENS:-256}
NUM_PREFIXES=${NUM_PREFIXES:-10}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4608}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
SEED=${SEED:-0}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REQUEST_SET=${REQUEST_SET:-benchmark_results/pr153-highprefix-ab-20260803/request_set.json}

case "$MODE" in
    native) SPEC_NAME=NPUOffloadingSpec ;;
    mapped) SPEC_NAME=MappedOffloadingSpec ;;
    *)
        echo "mode must be native or mapped" >&2
        exit 2
        ;;
esac

if ! [[ "$LIFECYCLE_INDEX" =~ ^[1-9][0-9]*$ ]]; then
    echo "lifecycle-index must be a positive integer" >&2
    exit 2
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
RUN_DIR=$(realpath -m "$RUN_DIR")
REQUEST_SET=$(realpath "$REQUEST_SET")
RUN_BASENAME=$(basename "$RUN_DIR")
if ! [[ "$RUN_BASENAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "run directory basename contains unsupported characters: $RUN_BASENAME" >&2
    exit 2
fi
RUN_ID="pr153-highprefix-$RUN_BASENAME"

if [[ -e "$RUN_DIR" ]]; then
    echo "run directory already exists: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"

docker inspect "$CONTAINER" >/dev/null
if [[ $(docker inspect -f '{{.State.Running}}' "$CONTAINER") != true ]]; then
    echo "container is not running: $CONTAINER" >&2
    exit 1
fi

VLLM_COMMIT=$(docker inspect -f \
    '{{index .Config.Labels "dev.vllm-hust.vllm-sha"}}' "$CONTAINER")
VLLM_SOURCE=$(docker inspect -f \
    '{{index .Config.Labels "dev.vllm-hust.vllm-repo"}}' "$CONTAINER")
VLLM_HEAD_TREE=$(git -C "$VLLM_SOURCE" rev-parse HEAD^{tree})
VLLM_DIFF_SHA256=$(git -C "$VLLM_SOURCE" diff HEAD | sha256sum | cut -d' ' -f1)
VLLM_DIRTY=$(
    if [[ -n $(git -C "$VLLM_SOURCE" status --porcelain --untracked-files=no) ]]; then
        printf true
    else
        printf false
    fi
)
ASCEND_SOURCE=$(docker inspect -f \
    '{{range .Mounts}}{{if eq .Destination "/workspace/vllm-ascend"}}{{.Source}}{{end}}{{end}}' \
    "$CONTAINER")
VLLM_ASCEND_COMMIT=$(git -C "$ASCEND_SOURCE" rev-parse HEAD)
VLLM_ASCEND_HEAD_TREE=$(git -C "$ASCEND_SOURCE" rev-parse HEAD^{tree})
VLLM_ASCEND_DIFF_SHA256=$(git -C "$ASCEND_SOURCE" diff HEAD | sha256sum | cut -d' ' -f1)
VLLM_ASCEND_DIRTY=$(
    if [[ -n $(git -C "$ASCEND_SOURCE" status --porcelain --untracked-files=no) ]]; then
        printf true
    else
        printf false
    fi
)
CONTAINER_IMAGE_REF=$(docker inspect -f \
    '{{index .Config.Labels "dev.vllm-hust.image-ref"}}' "$CONTAINER")
CONTAINER_IMAGE_ID=$(docker inspect -f \
    '{{index .Config.Labels "dev.vllm-hust.image-id"}}' "$CONTAINER")
MODEL_REPO_HOST=$(docker inspect -f \
    '{{index .Config.Labels "dev.vllm-hust.model-repo"}}' "$CONTAINER")
MODEL_CONFIG_SHA256=$(sha256sum "$MODEL_REPO_HOST/config.json" | cut -d' ' -f1)

NPU_PROCESS_STATE=$(npu-smi info -t proc-mem -i "$PHYSICAL_DEVICE_ID" -c 0)
if ! grep -q 'No process in device' <<<"$NPU_PROCESS_STATE"; then
    printf '%s\n' "$NPU_PROCESS_STATE" >&2
    echo "physical NPU $PHYSICAL_DEVICE_ID has an active process" >&2
    exit 1
fi
npu-smi info >"$RUN_DIR/npu-before-server.txt"

cat >"$RUN_DIR/run-config.json" <<EOF
{
  "mode": "$MODE",
  "spec_name": "$SPEC_NAME",
  "spec_module_path": "vllm_ascend.kv_offload.npu",
  "lifecycle_index": $LIFECYCLE_INDEX,
  "container": "$CONTAINER",
  "container_image_ref": "$CONTAINER_IMAGE_REF",
  "container_image_id": "$CONTAINER_IMAGE_ID",
  "vllm_commit": "$VLLM_COMMIT",
  "vllm_head_tree": "$VLLM_HEAD_TREE",
  "vllm_diff_sha256": "$VLLM_DIFF_SHA256",
  "vllm_dirty": $VLLM_DIRTY,
  "vllm_ascend_commit": "$VLLM_ASCEND_COMMIT",
  "vllm_ascend_head_tree": "$VLLM_ASCEND_HEAD_TREE",
  "vllm_ascend_diff_sha256": "$VLLM_ASCEND_DIFF_SHA256",
  "vllm_ascend_dirty": $VLLM_ASCEND_DIRTY,
  "physical_device_id": "$PHYSICAL_DEVICE_ID",
  "port": $PORT,
  "model": "$MODEL_NAME",
  "model_path": "$MODEL_PATH",
  "model_repo_host": "$MODEL_REPO_HOST",
  "model_config_sha256": "$MODEL_CONFIG_SHA256",
  "device_kv_bytes": $DEVICE_KV_BYTES,
  "cpu_kv_bytes": $CPU_KV_BYTES,
  "request_count": $REQUEST_COUNT,
  "request_rate": $REQUEST_RATE,
  "prefix_tokens": $PREFIX_TOKENS,
  "suffix_tokens": $SUFFIX_TOKENS,
  "output_tokens": $OUTPUT_TOKENS,
  "num_prefixes": $NUM_PREFIXES,
  "max_model_len": $MAX_MODEL_LEN,
  "max_num_seqs": $MAX_NUM_SEQS,
  "graph_mode": true,
  "prefix_caching": true,
  "prefix_caching_hash_algo": "sha256",
  "seed": $SEED
}
EOF

monitor_resources() {
    while true; do
        printf 'timestamp_ns=%s\n' "$(date +%s%N)"
        npu-smi info -t usages -i "$PHYSICAL_DEVICE_ID" -c 0 || true
        npu-smi info -t proc-mem -i "$PHYSICAL_DEVICE_ID" -c 0 || true
        docker stats --no-stream --format \
            'container={{.Name}} mem={{.MemUsage}} cpu={{.CPUPerc}}' \
            "$CONTAINER" || true
        sleep 1
    done
}

monitor_resources >"$RUN_DIR/resource-samples.log" 2>&1 &
MONITOR_PID=$!
cleanup_monitor() {
    if kill -0 "$MONITOR_PID" 2>/dev/null; then
        kill "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
}
trap cleanup_monitor EXIT

set +e
docker exec -i \
    -e MODE="$MODE" \
    -e SPEC_NAME="$SPEC_NAME" \
    -e RUN_ID="$RUN_ID" \
    -e PORT="$PORT" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e MODEL_PATH="$MODEL_PATH" \
    -e DEVICE_KV_BYTES="$DEVICE_KV_BYTES" \
    -e CPU_KV_BYTES="$CPU_KV_BYTES" \
    -e REQUEST_COUNT="$REQUEST_COUNT" \
    -e REQUEST_RATE="$REQUEST_RATE" \
    -e PREFIX_TOKENS="$PREFIX_TOKENS" \
    -e SUFFIX_TOKENS="$SUFFIX_TOKENS" \
    -e OUTPUT_TOKENS="$OUTPUT_TOKENS" \
    -e NUM_PREFIXES="$NUM_PREFIXES" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    -e SEED="$SEED" \
    "$CONTAINER" bash -s >"$RUN_DIR/controller.log" 2>&1 <<'CONTAINER_SCRIPT'
set -euo pipefail

WORK_DIR="/tmp/$RUN_ID"
if [[ -e "$WORK_DIR" ]]; then
    echo "container run directory already exists: $WORK_DIR" >&2
    exit 1
fi
mkdir -p "$WORK_DIR/trace"

cleanup_server() {
    if [[ -n ${SERVER_PID:-} ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    # The multiprocess API server can leave its EngineCore child alive after
    # the parent exits. This container is task-specific and runs one lifecycle
    # at a time, so terminate only this task's named engine process before the
    # host records the post-run NPU state.
    pkill -TERM -f '^VLLM::EngineCore' 2>/dev/null || true
    for _ in $(seq 1 30); do
        pgrep -f '^VLLM::EngineCore' >/dev/null || return 0
        sleep 1
    done
    pkill -KILL -f '^VLLM::EngineCore' 2>/dev/null || true
}
trap cleanup_server EXIT

KV_TRANSFER_CONFIG=$(cat <<EOF
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "$SPEC_NAME",
    "spec_module_path": "vllm_ascend.kv_offload.npu",
    "cpu_bytes_to_use": $CPU_KV_BYTES,
    "block_size": 128,
    "eviction_policy": "lru"
  }
}
EOF
)

export PYTHONPATH=/workspace/vllm:/workspace/vllm-ascend
export VLLM_KV_TRANSFER_TRACE_DIR="$WORK_DIR/trace"
export VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1
export VLLM_ASCEND_TORCH_PREFLIGHT=0
export VLLM_ASCEND_DISABLE_TOP_K_TOP_P_CUSTOM_OP=1

python -u -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --disable-uvicorn-access-log \
    --dtype float16 \
    --served-model-name "$MODEL_NAME" \
    --generation-config vllm \
    --max-model-len "$MAX_MODEL_LEN" \
    --kv-cache-memory-bytes "$DEVICE_KV_BYTES" \
    --enable-prefix-caching \
    --prefix-caching-hash-algo sha256 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    >"$WORK_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" >"$WORK_DIR/server.pid"

ready=false
for _ in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "server exited before becoming healthy" >&2
        tail -200 "$WORK_DIR/server.log" >&2
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done
if [[ "$ready" != true ]]; then
    echo "server did not become healthy within 300 seconds" >&2
    exit 1
fi

curl -fsS "http://127.0.0.1:$PORT/metrics" >"$WORK_DIR/metrics_before.prom"

python -u -m vllm.entrypoints.cli.main bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/completions \
    --model "$MODEL_NAME" \
    --tokenizer "$MODEL_PATH" \
    --dataset-name prefix_repetition \
    --num-prompts "$REQUEST_COUNT" \
    --prefix-repetition-prefix-len "$PREFIX_TOKENS" \
    --prefix-repetition-suffix-len "$SUFFIX_TOKENS" \
    --prefix-repetition-num-prefixes "$NUM_PREFIXES" \
    --prefix-repetition-output-len "$OUTPUT_TOKENS" \
    --request-rate "$REQUEST_RATE" \
    --seed "$SEED" \
    --disable-tqdm \
    --save-result \
    --save-detailed \
    --result-dir "$WORK_DIR" \
    --result-filename result.json \
    --ignore-eos \
    --request-id-prefix highprefix- \
    >"$WORK_DIR/client.log" 2>&1

curl -fsS "http://127.0.0.1:$PORT/metrics" >"$WORK_DIR/metrics_after.prom"
cleanup_server
SERVER_PID=
CONTAINER_SCRIPT
RUN_RC=$?
set -e

cleanup_monitor
trap - EXIT

docker cp "$CONTAINER:/tmp/$RUN_ID/." "$RUN_DIR/" >/dev/null
npu-smi info >"$RUN_DIR/npu-after-shutdown.txt"

if [[ $RUN_RC -ne 0 ]]; then
    echo "lifecycle failed; see $RUN_DIR/controller.log" >&2
    exit "$RUN_RC"
fi

python3 "$SCRIPT_DIR/collect_transfer_events.py" \
    "$RUN_DIR/trace" "$RUN_DIR/transfer_events.jsonl" \
    >"$RUN_DIR/collect.log" 2>&1
python3 "$SCRIPT_DIR/convert_benchmark_result.py" \
    "$RUN_DIR/result.json" "$RUN_DIR/raw_requests.jsonl" \
    --request-set "$REQUEST_SET" \
    >"$RUN_DIR/convert.log" 2>&1

cp "$REQUEST_SET" "$RUN_DIR/request_set.json"
printf 'completed %s lifecycle %s in %s\n' \
    "$MODE" "$LIFECYCLE_INDEX" "$RUN_DIR"
