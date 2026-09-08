#!/usr/bin/env bash
set -euo pipefail

readonly VIEW_DIR="/data2/zrj2025/uav-results/tensorboard_views/reward_redesign"
readonly PYTHON="/data2/zrj2025/.conda/envs/uavmec312/bin/python"
readonly PID_FILE="${VIEW_DIR}/tensorboard.pid"
readonly LOG_FILE="${VIEW_DIR}/tensorboard.log"
readonly PORT="16007"

mkdir -p "${VIEW_DIR}"

if [[ -f "${PID_FILE}" ]]; then
    existing_pid="$(<"${PID_FILE}")"
    if kill -0 "${existing_pid}" 2>/dev/null \
        && ps -p "${existing_pid}" -o args= | grep -Fq -- "--logdir ${VIEW_DIR}"; then
        exit 0
    fi
fi

nohup "${PYTHON}" -m tensorboard.main \
    --logdir "${VIEW_DIR}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --reload_interval 30 \
    >>"${LOG_FILE}" 2>&1 </dev/null &
echo "$!" >"${PID_FILE}"
