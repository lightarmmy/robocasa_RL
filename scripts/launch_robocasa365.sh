# Copyright (C) 2026 Xiaomi Corporation.
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BASE_PORT="${BASE_PORT:-10086}"
SERVER_ADDR="${SERVER_ADDR:-127.0.0.1}"
PYTHON="${PYTHON:-python}"
MODEL_DEFAULT="XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"

NUM_PORTS="${1:-}"
LOG_PATH="${2:-}"
MODEL_PATH="${3:-${MODEL_PATH:-${MODEL_DEFAULT}}}"
shift $(( $# >= 3 ? 3 : $# ))

if [[ -z "${NUM_PORTS}" || -z "${LOG_PATH}" ]]; then
    echo "Usage: bash scripts/launch_robocasa365.sh <num_ports> <log_path> [model_path]" >&2
    echo "       [extra evaluator args...]" >&2
    echo "Example: bash scripts/launch_robocasa365.sh 8 ./eval_results/robocasa365 /path/to/checkpoint" >&2
    exit 1
fi
if ! [[ "${NUM_PORTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: num_ports must be a positive integer." >&2
    exit 1
fi

# Set CONDA_ENV when the client should activate a dedicated simulator environment.
if [[ -n "${CONDA_ENV:-}" ]]; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
QUEUE_DIR="${QUEUE_DIR:-${LOG_PATH}/scheduler/${RUN_ID}}"
SPLIT="${SPLIT:-pretrain}"
TASK_SET="${TASK_SET:-target50}"
NUM_TRIALS="${NUM_TRIALS:-50}"
REPLAN_STEPS="${REPLAN_STEPS:-16}"
OBS_HISTORY="${OBS_HISTORY:-4}"
OBS_INTERVAL="${OBS_INTERVAL:-2}"
SEED="${SEED:-7}"
CROP_RATIO="${CROP_RATIO:-0.95}"
CAMERA_SAMPLING_INTERVAL="${CAMERA_SAMPLING_INTERVAL:-1}"

"${PYTHON}" -u "${REPO_ROOT}/eval_robocasa365/dynamic_eval.py" init \
    --queue-dir "${QUEUE_DIR}" -- \
    "$@" \
    --model-path "${MODEL_PATH}" \
    --server-addr "${SERVER_ADDR}" \
    --split "${SPLIT}" \
    --task-set "${TASK_SET}" \
    --num-trials "${NUM_TRIALS}" \
    --replan-steps "${REPLAN_STEPS}" \
    --obs-history "${OBS_HISTORY}" \
    --obs-interval "${OBS_INTERVAL}" \
    --seed "${SEED}" \
    --crop-ratio "${CROP_RATIO}" \
    --camera-sampling-interval "${CAMERA_SAMPLING_INTERVAL}" \
    --save-root-dir "${LOG_PATH}" \
    --run-id "${RUN_ID}"

worker_pids=()
cleanup_workers() {
    if (( ${#worker_pids[@]} > 0 )); then
        kill "${worker_pids[@]}" 2>/dev/null || true
        wait "${worker_pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup_workers EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

WORKER_MAX_JOBS="${WORKER_MAX_JOBS:-1}"
WORKER_MAX_ATTEMPTS="${WORKER_MAX_ATTEMPTS:-3}"
if ! [[ "${WORKER_MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WORKER_MAX_JOBS must be a positive integer, got: ${WORKER_MAX_JOBS}" >&2
    exit 1
fi
if ! [[ "${WORKER_MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "WORKER_MAX_ATTEMPTS must be a positive integer, got: ${WORKER_MAX_ATTEMPTS}" >&2
    exit 1
fi

for ((index = 0; index < NUM_PORTS; index++)); do
    port=$((BASE_PORT + index))
    worker_env=("PYTHONUNBUFFERED=1")
    # When multiple RoboCasa clients render concurrently, bind each EGL
    # context to a distinct visible GPU to avoid native MuJoCo aborts.
    if [[ "${EGL_DEVICE_PER_WORKER:-0}" == "1" ]]; then
        worker_env+=("MUJOCO_EGL_DEVICE_ID=${index}")
    fi
    worker_env+=("PYTHONFAULTHANDLER=1")
    (
        # A RoboCasa EGL context can survive env.close() and eventually abort
        # in read_pixels.  Restart the Python worker after a bounded number of
        # episodes, which resets native OpenGL/MuJoCo global state as well.
        while compgen -G "${QUEUE_DIR}/pending/*.json" > /dev/null; do
            worker_command=(
                "${PYTHON}" -u "${REPO_ROOT}/eval_robocasa365/dynamic_eval.py" worker
                --queue-dir "${QUEUE_DIR}"
                --worker-id "gpu-${index}"
                --server-addr "${SERVER_ADDR}"
                --server-port "${port}"
                --max-jobs "${WORKER_MAX_JOBS}"
            )
            if env "${worker_env[@]}" "${worker_command[@]}"; then
                continue
            else
                worker_status=$?
            fi
            echo "Worker gpu-${index} exited with status ${worker_status}; recovering stranded rollout" >&2
            if ! "${PYTHON}" -u "${REPO_ROOT}/eval_robocasa365/dynamic_eval.py" recover \
                --queue-dir "${QUEUE_DIR}" \
                --worker-id "gpu-${index}" \
                --max-attempts "${WORKER_MAX_ATTEMPTS}"; then
                exit "${worker_status}"
            fi
        done
    ) >"${QUEUE_DIR}/logs/worker-${index}.log" 2>&1 &
    worker_pids+=("$!")
done

worker_status=0
for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        worker_status=1
    fi
done

if ! "${PYTHON}" -u "${REPO_ROOT}/eval_robocasa365/dynamic_eval.py" merge \
    --queue-dir "${QUEUE_DIR}"; then
    worker_status=1
fi

if (( worker_status != 0 )); then
    echo "Evaluation did not complete. Inspect ${QUEUE_DIR}/logs and ${QUEUE_DIR}/errors." >&2
    exit "${worker_status}"
fi

echo "Evaluation complete. Results: ${LOG_PATH}/${RUN_ID}"
